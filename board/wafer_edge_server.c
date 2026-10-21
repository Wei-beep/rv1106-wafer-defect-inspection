// =====================================================================
// RV1106 边缘推理 + Web(SSE) 监控一体化服务  wafer_edge_server.c
//   TCP 推理  : 8080  (收 RFNS 12356B, 回 RFNR 44B)
//   HTTP 看板 : 8081
//     GET /                     仪表盘 (双语, SSE 实时)
//     GET /api/device/status    一次性 JSON 状态
//     GET /api/stream           SSE 长连接 (infer / batch 事件)
// SSE 事件:
//   infer: 每帧推理结果 (含次选类别, NPU/e2e 时延)
//   batch: 批量汇总 (每类数量, 主导缺陷, 工艺建议, 整体评价)
// 零第三方依赖.
// 编译:
//   arm-rockchip830-linux-uclibcgnueabihf-gcc wafer_edge_server.c -o wafer_edge \
//     -I./ -L<librknnmrt目录> -lrknnmrt -lm -lpthread -O2
// =====================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#include "rknn_api.h"

#define IMG_BYTES   (64*64*3)
#define FEAT_BYTES  (16*4)
#define REQ_BYTES   (4+IMG_BYTES+FEAT_BYTES)
#define RSP_BYTES   (4+4+4+8*4)
#define NC 8
#define RECENT_N 32
#define SSE_MAX 8
#define BATCH_GAP_MS 3000.0

static const char* CLASS_NAMES[NC] = {
    "Center","Donut","Edge-Loc","Edge-Ring","Loc","Near-full","Random","Scratch"
};
/* 工艺建议映射 (与 Qt 端一致) */
static const char* SUGGEST[NC] = {
    "光刻/对准",          /* Center    */
    "薄膜沉积/刻蚀",       /* Donut     */
    "边缘曝光",            /* Edge-Loc  */
    "边缘刻蚀/沉积",       /* Edge-Ring */
    "局部工艺偏差",        /* Loc       */
    "整片工艺异常",        /* Near-full */
    "随机污染/颗粒",       /* Random    */
    "机械划伤/搬运"        /* Scratch   */
};

/* ---------------- 共享统计 ---------------- */
typedef struct {
    pthread_mutex_t m;
    long   total;
    double npuSum;
    double e2eSum;
    int    lastPred;
    float  lastConf;
    double bootMs;
    struct { int pred; float conf; double npu; double e2e; } ring[RECENT_N];
    int ringHead;
} Stats;
static Stats G;

/* ---------------- 批量状态 ---------------- */
typedef struct {
    int    id;
    int    count;
    int    classCount[NC];
    double sumNpu;
    double sumE2e;
    double startTime;
    double firstFrameMs;
} BatchState;
static BatchState B;
static double lastReqTime = 0;
static pthread_mutex_t batch_m;

/* ---------------- SSE 客户端 ---------------- */
static int sse_fds[SSE_MAX];
static pthread_mutex_t sse_m;

static void sse_init(void){
    pthread_mutex_init(&sse_m,0);
    for(int i=0;i<SSE_MAX;i++) sse_fds[i]=-1;
}
static void sse_add(int fd){
    pthread_mutex_lock(&sse_m);
    for(int i=0;i<SSE_MAX;i++) if(sse_fds[i]<0){sse_fds[i]=fd;break;}
    pthread_mutex_unlock(&sse_m);
}
static void sse_broadcast(const char* data){
    char buf[2048];
    int n=snprintf(buf,sizeof(buf),"data: %s\n\n",data);
    pthread_mutex_lock(&sse_m);
    for(int i=0;i<SSE_MAX;i++){
        int fd=sse_fds[i];
        if(fd<0) continue;
        int w=send(fd,buf,n,MSG_NOSIGNAL);
        if(w<=0){ close(fd); sse_fds[i]=-1; }
    }
    pthread_mutex_unlock(&sse_m);
}
static void* sse_heartbeat(void* arg){
    (void)arg;
    const char* hb=": heartbeat\n\n";
    while(1){
        sleep(10);
        pthread_mutex_lock(&sse_m);
        for(int i=0;i<SSE_MAX;i++){
            int fd=sse_fds[i];
            if(fd<0) continue;
            if(send(fd,hb,strlen(hb),MSG_NOSIGNAL)<=0){ close(fd); sse_fds[i]=-1; }
        }
        pthread_mutex_unlock(&sse_m);
    }
    return NULL;
}

static double now_ms(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1000.0 + t.tv_nsec/1e6;
}
static void* read_file(const char*p,int*n){
    FILE*f=fopen(p,"rb"); if(!f)return NULL;
    fseek(f,0,SEEK_END); *n=ftell(f); fseek(f,0,SEEK_SET);
    void*b=malloc(*n); fread(b,1,*n,f); fclose(f); return b;
}
static int8_t q1(float x,float s,int32_t z){
    float v=x/s+z; if(v>127)v=127; if(v<-128)v=-128; return(int8_t)(v+0.5f);
}
static int recv_n(int fd,void*buf,int n){
    char*p=(char*)buf; int left=n;
    while(left>0){int r=recv(fd,p,left,0); if(r<=0)return -1; p+=r; left-=r;}
    return n;
}
static int send_n(int fd,const void*buf,int n){
    const char*p=(const char*)buf; int left=n;
    while(left>0){int s=send(fd,p,left,0); if(s<=0)return -1; p+=s; left-=s;}
    return n;
}

/* 发送批量汇总事件 */
static void finalizeBatch(void){
    if(B.count==0) return;
    int dominant=0, maxC=0;
    for(int k=0;k<NC;k++) if(B.classCount[k]>maxC){maxC=B.classCount[k];dominant=k;}
    double dur = now_ms()-B.startTime;
    double avgNpu = B.sumNpu/B.count;
    double avgE2e = B.sumE2e/B.count;
    /* normal = Near-full + Random 视为正常片 */
    int normal = B.classCount[5]+B.classCount[6];
    float domPct = (float)maxC/B.count*100.0f;
    const char* eval = (domPct>=60.0f) ? "本批缺陷集中，建议停线排查工艺"
                                          : "本批缺陷分散，建议持续监控";
    char j[2048];
    int pos=snprintf(j,sizeof(j),
      "{\"type\":\"batch\",\"batchId\":%d,\"count\":%d,\"duration\":%.1f,"
      "\"avgNpu\":%.2f,\"avgE2e\":%.2f,\"firstFrame\":%.2f,"
      "\"normal\":%d,\"dominant\":\"%s\",\"domPct\":%.1f,"
      "\"suggestion\":\"%s\",\"evaluation\":\"%s\",\"classes\":{",
      B.id, B.count, dur, avgNpu, avgE2e, B.firstFrameMs,
      normal, CLASS_NAMES[dominant], domPct,
      SUGGEST[dominant], eval);
    int first=1;
    for(int k=0;k<NC;k++){
        if(B.classCount[k]==0) continue;
        pos+=snprintf(j+pos,sizeof(j)-pos,"%s\"%s\":%d",
            first?"":",", CLASS_NAMES[k], B.classCount[k]);
        first=0;
    }
    snprintf(j+pos,sizeof(j)-pos,"}}");
    sse_broadcast(j);
    printf("[BATCH] #%d count=%d dominant=%s(%.1f%%) avgNpu=%.2f avgE2e=%.2f\n",
           B.id, B.count, CLASS_NAMES[dominant], domPct, avgNpu, avgE2e);
}

static void startNewBatch(double now){
    B.id++;
    B.count=0;
    memset(B.classCount,0,sizeof(B.classCount));
    B.sumNpu=0; B.sumE2e=0;
    B.startTime=now; B.firstFrameMs=0;
}

/* 记录单帧 + 推送 infer 事件 */
static void recordInfer(int pred,float conf,double npu,double e2e,
                        int secondPred,float secondConf){
    pthread_mutex_lock(&G.m);
    G.total++; G.npuSum+=npu; G.e2eSum+=e2e;
    G.lastPred=pred; G.lastConf=conf;
    G.ring[G.ringHead].pred=pred; G.ring[G.ringHead].conf=conf;
    G.ring[G.ringHead].npu=npu;   G.ring[G.ringHead].e2e=e2e;
    G.ringHead=(G.ringHead+1)%RECENT_N;
    double avgNpu=G.npuSum/G.total, avgE2e=G.e2eSum/G.total;
    pthread_mutex_unlock(&G.m);

    pthread_mutex_lock(&batch_m);
    double now=now_ms();
    if(lastReqTime>0 && now-lastReqTime>BATCH_GAP_MS){
        finalizeBatch();
        startNewBatch(now);
    }
    if(B.count==0){ B.startTime=now; B.id = B.id>0?B.id:1; }
    B.count++;
    B.classCount[pred]++;
    B.sumNpu+=npu; B.sumE2e+=e2e;
    if(B.firstFrameMs==0) B.firstFrameMs=e2e;
    int batchId=B.id;
    lastReqTime=now;
    pthread_mutex_unlock(&batch_m);

    char j[512];
    snprintf(j,sizeof(j),
      "{\"type\":\"infer\",\"class\":\"%s\",\"conf\":%.3f,\"npu\":%.2f,\"e2e\":%.2f,"
      "\"total\":%ld,\"avgNpu\":%.2f,\"avgE2e\":%.2f,"
      "\"secondClass\":\"%s\",\"secondConf\":%.3f,\"batchId\":%d}",
      CLASS_NAMES[pred], conf, npu, e2e,
      G.total, avgNpu, avgE2e,
      CLASS_NAMES[secondPred], secondConf, batchId);
    sse_broadcast(j);
}

/* ---------------- RKNN ---------------- */
static rknn_context ctx;
static rknn_tensor_attr ina[2], outa;
static rknn_tensor_mem *inm[2]={0}, *outm=0;
static int ii, fi;

static int rknn_setup(const char* modelPath){
    int rc,mlen=0; void*model=read_file(modelPath,&mlen);
    if((rc=rknn_init(&ctx,model,mlen,0,NULL))<0){printf("rknn_init=%d\n",rc);return -1;}
    rknn_input_output_num ion; memset(&ion,0,sizeof(ion));
    rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&ion,sizeof(ion));
    for(uint32_t i=0;i<ion.n_input;i++){
        memset(&ina[i],0,sizeof(ina[i])); ina[i].index=i;
        rknn_query(ctx,RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR,&ina[i],sizeof(ina[i]));
        uint32_t al=ina[i].size_with_stride?ina[i].size_with_stride:ina[i].size;
        inm[i]=rknn_create_mem(ctx,al);
        rknn_set_io_mem(ctx,inm[i],&ina[i]);
    }
    memset(&outa,0,sizeof(outa)); outa.index=0;
    rknn_query(ctx,RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR,&outa,sizeof(outa));
    outm=rknn_create_mem(ctx,outa.size_with_stride?outa.size_with_stride:outa.size);
    rknn_set_io_mem(ctx,outm,&outa);
    ii = ina[0].size>=IMG_BYTES?0:1; fi=1-ii;
    printf("[INFO] RKNN ready, in0=%s in1=%s out=%s\n",
           ina[0].name, ina[1].name, outa.name);
    return 0;
}
static int do_infer(const unsigned char* img, const float* feat,
                    int* pred, float* conf, float* logits,
                    int* secondPred, float* secondConf){
    memcpy(inm[ii]->virt_addr, img, IMG_BYTES);
    int8_t* fd=(int8_t*)inm[fi]->virt_addr;
    for(int j=0;j<16;j++) fd[j]=q1(feat[j], ina[fi].scale, ina[fi].zp);
    if(rknn_run(ctx,NULL)<0) return -1;
    int8_t* oq=(int8_t*)outm->virt_addr;
    float mx=-1e9f, second=-1e9f; *pred=0; *secondPred=0;
    for(int k=0;k<NC;k++){
        logits[k]=(oq[k]-outa.zp)*outa.scale;
        if(logits[k]>mx){second=mx;*secondPred=*pred;mx=logits[k];*pred=k;}
        else if(logits[k]>second){second=logits[k];*secondPred=k;}
    }
    float se=0,tmp[NC];
    for(int k=0;k<NC;k++){tmp[k]=expf(logits[k]-mx); se+=tmp[k];}
    *conf=tmp[*pred]/se;
    *secondConf=tmp[*secondPred]/se;
    return 0;
}

/* ---------------- TCP 推理服务 (8080) ---------------- */
static void* tcp_loop(void* arg){
    (void)arg;
    int lsn=socket(AF_INET,SOCK_STREAM,0); int opt=1;
    setsockopt(lsn,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
    struct sockaddr_in a; memset(&a,0,sizeof(a));
    a.sin_family=AF_INET; a.sin_addr.s_addr=htonl(INADDR_ANY); a.sin_port=htons(8080);
    if(bind(lsn,(struct sockaddr*)&a,sizeof(a))<0){perror("bind 8080");return NULL;}
    listen(lsn,4);
    printf("[INFO] TCP infer on :8080\n");
    char req[REQ_BYTES];
    while(1){
        int cfd=accept(lsn,NULL,NULL); if(cfd<0)continue;
        while(1){
            char magic[4];
            if(recv_n(cfd,magic,4)!=4 || memcmp(magic,"RFNS",4)!=0) break;
            if(recv_n(cfd,req,REQ_BYTES-4)!=REQ_BYTES-4) break;
            double tStart=now_ms();  /* 全量收包完成, 开始计板端处理时延 */
            int pred,secondPred; float conf,secondConf,lg[NC];
            double t0=now_ms();
            if(do_infer((unsigned char*)req,(float*)(req+IMG_BYTES),
                         &pred,&conf,lg,&secondPred,&secondConf)<0) break;
            double npu=now_ms()-t0;
            char rsp[RSP_BYTES]; char*rp=rsp;
            memcpy(rp,"RFNR",4); rp+=4;
            int32_t p32=pred; memcpy(rp,&p32,4); rp+=4;
            memcpy(rp,&conf,4); rp+=4;
            for(int k=0;k<NC;k++){memcpy(rp,&lg[k],4); rp+=4;}
            send_n(cfd,rsp,RSP_BYTES);
            double e2e=now_ms()-tStart;  /* 收包完成→回包发送完成 */
            recordInfer(pred,conf,npu,e2e,secondPred,secondConf);
        }
        close(cfd);
    }
    return NULL;
}

/* ---------------- HTTP + SSE 看板 (8081) ---------------- */
static const char* HTML =
"<!doctype html><html><head><meta charset=utf-8>"
"<meta name=viewport content='width=device-width,initial-scale=1'>"
"<title>Wafer Edge Monitor</title><style>"
"body{background:#0f1419;color:#e6e6e6;font-family:Segoe UI,Arial;margin:0;padding:20px}"
".topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}"
"h1{font-size:18px;margin:0;color:#4fc3f7}"
"#btnLang{background:#1a222b;color:#4fc3f7;border:1px solid #2a3540;border-radius:6px;"
"padding:5px 12px;cursor:pointer;font-size:12px}"
".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:14px}"
".card{background:#1a222b;border:1px solid #2a3540;border-radius:8px;padding:12px}"
".v{font-size:22px;font-weight:700;color:#fff}.l{font-size:11px;color:#8aa0b3;margin-top:3px}"
"table{width:100%;border-collapse:collapse;margin-top:8px;font-size:12px}"
"td,th{padding:5px 7px;border-bottom:1px solid #26303a;text-align:left}"
".ok{color:#66bb6a}.bad{color:#ef5350}.warn{color:#ffa726}"
".lowconf{color:#ef5350;font-weight:700}"
".dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#66bb6a;"
"margin-right:6px;animation:pulse 1.2s infinite}"
"@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}"
".tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px}"
".tag-batch{background:#1565c0;color:#fff}.tag-single{background:#6a1b9a;color:#fff}"
".tag-idle{background:#424242;color:#bdbdbd}"
".batch-card{background:#151c25;border:1px solid #2a3540;border-radius:8px;margin-top:10px;overflow:hidden}"
".batch-head{padding:10px 14px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;"
"background:#1a222b;font-weight:600}"
".batch-head:hover{background:#243040}"
".batch-body{padding:12px 14px;display:none;font-size:12px;line-height:1.8}"
".batch-body.open{display:block}"
".batch-class{display:inline-block;margin-right:16px}"
".suggestion{color:#ffa726;font-weight:600}"
".eval{color:#ef5350;font-weight:600}"
".sec{color:#8aa0b3}"
"</style></head><body>"
"<div class='topbar'>"
"<h1><span class='dot'></span><span id='title'>RV1106 晶圆缺陷检测 · 边缘监控 (SSE 实时)</span></h1>"
"<button id='btnLang' onclick='toggleLang()'>EN</button>"
"</div>"
"<div class='grid'>"
"<div class='card'><div class='v' id='dev'>--</div><div class='l' id='lblStatus'>设备状态</div></div>"
"<div class='card'><div class='v' id='mode'><span class='tag tag-idle' id='modeTag'>Idle</span></div>"
"<div class='l' id='lblMode'>运行模式</div></div>"
"<div class='card'><div class='v' id='tot'>0</div><div class='l' id='lblTotal'>累计推理帧数</div></div>"
"<div class='card'><div class='v' id='npu'>--</div><div class='l' id='lblNpu'>平均 NPU 时延 (ms)</div></div>"
"<div class='card'><div class='v' id='e2e'>--</div><div class='l' id='lblE2e'>平均端到端时延 (ms)</div></div>"
"<div class='card'><div class='v' id='fps'>--</div><div class='l' id='lblFps'>稳态吞吐量 (帧/s)</div></div>"
"<div class='card'><div class='v' id='first'>--</div><div class='l' id='lblFirst'>首帧时延 (ms)</div></div>"
"<div class='card'><div class='v' id='last' style='font-size:16px'>--</div>"
"<div class='l' id='lblLast'>最近一次判定</div></div>"
"<div class='card'><div class='v' id='input' style='font-size:14px'>--</div>"
"<div class='l' id='lblInput'>输入类型 / 数量</div></div>"
"</div>"
"<div id='batchArea'></div>"
"<div class='card'><b id='lblStream'>实时推理流 (SSE 推送)</b>"
"<span style='float:right;font-size:11px;color:#8aa0b3' id='lblHint'>置信度 &lt; 0.50 标红</span>"
"<table><thead><tr>"
"<th>#</th><th id='thClass'>类别</th><th id='thConf'>置信度</th>"
"<th id='thSec'>次选类别</th><th id='thSecConf'>次选置信度</th>"
"<th id='thNpu'>NPU ms</th><th id='thE2e'>E2E ms</th><th id='thFlag'>标记</th>"
"</tr></thead><tbody id='tb'></tbody></table></div>"
"<script>"
"const LOW_TH=0.50;"
"const T={zh:{title:'RV1106 晶圆缺陷检测 · 边缘监控 (SSE 实时)',status:'设备状态',mode:'运行模式',"
"total:'累计推理帧数',npu:'平均 NPU 时延 (ms)',e2e:'平均端到端时延 (ms)',fps:'稳态吞吐量 (帧/s)',"
"first:'首帧时延 (ms)',last:'最近一次判定',input:'输入类型 / 数量',"
"stream:'实时推理流 (SSE 推送)',hint:'置信度 < 0.50 标红',cls:'类别',conf:'置信度',"
"sec:'次选类别',secConf:'次选置信度',npuCol:'NPU ms',e2eCol:'E2E ms',flag:'标记',"
"low:'低置信',batch:'批量处理',single:'单次推理',idle:'空闲',online:'在线',reconn:'重连中',"
"unknown:'未知',batchTitle:'批次 #',count:'片',duration:'耗时',avgNpu:'平均NPU',avgE2e:'平均E2E',"
"firstFrame:'首帧',normal:'正常片',dominant:'主导缺陷',suggest:'建议排查',eval:'整体评价',"
"classes:'缺陷分布',expand:'双击展开',collapse:'收起'},"
"en:{title:'RV1106 Wafer Defect Inspection · Edge Monitor (SSE Live)',status:'Status',mode:'Mode',"
"total:'Total Frames',npu:'Avg NPU Latency (ms)',e2e:'Avg E2E Latency (ms)',fps:'Throughput (fps)',"
"first:'First Frame (ms)',last:'Last Prediction',input:'Input Type / Count',"
"stream:'Live Inference Stream (SSE Push)',hint:'Confidence < 0.50 highlighted',cls:'Class',"
"conf:'Confidence',sec:'2nd Class',secConf:'2nd Conf',npuCol:'NPU ms',e2eCol:'E2E ms',flag:'Flag',"
"low:'Low Conf',batch:'Batch',single:'Single',idle:'Idle',online:'Online',reconn:'Reconnecting',"
"unknown:'Unknown',batchTitle:'Batch #',count:'frames',duration:'duration',avgNpu:'avg NPU',"
"avgE2e:'avg E2E',firstFrame:'1st frame',normal:'Normal',dominant:'Dominant',suggest:'Check',"
"eval:'Evaluation',classes:'Distribution',expand:'double-click to expand',collapse:'collapse'}};"
"let lang=localStorage.getItem('wafer_lang')||'zh';"
"let curStatus='offline';"
"function t(k){return (T[lang]&&T[lang][k])?T[lang][k]:k;}"
"function applyLang(){"
"document.getElementById('title').textContent=t('title');"
"document.getElementById('lblStatus').textContent=t('status');"
"document.getElementById('lblMode').textContent=t('mode');"
"document.getElementById('lblTotal').textContent=t('total');"
"document.getElementById('lblNpu').textContent=t('npu');"
"document.getElementById('lblE2e').textContent=t('e2e');"
"document.getElementById('lblFps').textContent=t('fps');"
"document.getElementById('lblFirst').textContent=t('first');"
"document.getElementById('lblLast').textContent=t('last');"
"document.getElementById('lblInput').textContent=t('input');"
"document.getElementById('lblStream').textContent=t('stream');"
"document.getElementById('lblHint').textContent=t('hint');"
"document.getElementById('thClass').textContent=t('cls');"
"document.getElementById('thConf').textContent=t('conf');"
"document.getElementById('thSec').textContent=t('sec');"
"document.getElementById('thSecConf').textContent=t('secConf');"
"document.getElementById('thNpu').textContent=t('npuCol');"
"document.getElementById('thE2e').textContent=t('e2eCol');"
"document.getElementById('thFlag').textContent=t('flag');"
"document.getElementById('btnLang').textContent=lang==='zh'?'EN':'中';"
"document.getElementById('dev').textContent=t(curStatus);"
"document.getElementById('dev').className='v'+(curStatus==='online'?' ok':' bad');"
"renderRows();updateModeTag();renderBatches();}"
"function toggleLang(){lang=lang==='zh'?'en':'zh';localStorage.setItem('wafer_lang',lang);applyLang();}"
"let rows=[];let frameTimes=[];let batches=[];"
"function updateModeTag(){"
"const now=Date.now();let recent=frameTimes.filter(ts=>now-ts<3000).length;"
"let tag=document.getElementById('modeTag');"
"if(recent>=3){tag.textContent=t('batch');tag.className='tag tag-batch';}"
"else if(recent>=1){tag.textContent=t('single');tag.className='tag tag-single';}"
"else{tag.textContent=t('idle');tag.className='tag tag-idle';}}"
"function calcFps(){"
"const now=Date.now();let recent=frameTimes.filter(ts=>now-ts<5000).length;"
"return recent>0?(recent/5.0).toFixed(1):'--';}"
"function renderRows(){"
"let h='';rows.forEach((q,i)=>{"
"let low=q.conf<LOW_TH;"
"h+='<tr><td>'+(i+1)+'</td><td>'+q.class+'</td>';"
"h+='<td class='+(low?'lowconf':'')+'>'+q.conf.toFixed(3)+'</td>';"
"h+='<td class=sec>'+q.secondClass+'</td>';"
"h+='<td class=sec>'+q.secondConf.toFixed(3)+'</td>';"
"h+='<td>'+q.npu.toFixed(2)+'</td><td>'+q.e2e.toFixed(2)+'</td>';"
"h+='<td>'+(low?'<span class=lowconf>'+t('low')+'</span>':'')+'</td></tr>';});"
"document.getElementById('tb').innerHTML=h;}"
"function renderBatches(){"
"let h='';batches.forEach((b,idx)=>{"
"let clsHtml='';for(let c in b.classes){clsHtml+='<span class=batch-class>'+c+': <b>'+b.classes[c]+'</b> ('+(b.classes[c]/b.count*100).toFixed(1)+'%)</span>';}"
"h+='<div class=batch-card>';"
"h+='<div class=batch-head ondblclick=\"toggleBatch('+idx+')\">';"
"h+='<span>'+t('batchTitle')+b.batchId+' &nbsp; '+b.count+t('count')+' &nbsp; '+"
"t('dominant')+': '+b.dominant+' ('+b.domPct+'%)</span>';"
"h+='<span style=font-size:11px;color:#8aa0b3 id=bh'+idx+'>'+t('expand')+'</span></div>';"
"h+='<div class=batch-body id=bb'+idx+'>';"
"h+='<div><b>'+t('duration')+':</b> '+b.duration.toFixed(1)+'s &nbsp; <b>'+t('avgNpu')+':</b> '+b.avgNpu.toFixed(2)+'ms &nbsp; <b>'+t('avgE2e')+':</b> '+b.avgE2e.toFixed(2)+'ms &nbsp; <b>'+t('firstFrame')+':</b> '+b.firstFrame.toFixed(2)+'ms</div>';"
"h+='<div style=margin-top:6px><b>'+t('classes')+':</b><br>'+clsHtml+'</div>';"
"h+='<div style=margin-top:6px><b>'+t('normal')+':</b> '+b.normal+' ('+(b.normal/b.count*100).toFixed(1)+'%)</div>';"
"h+='<div class=suggestion style=margin-top:4px><b>'+t('suggest')+':</b> '+b.suggestion+'</div>';"
"h+='<div class=eval><b>'+t('eval')+':</b> '+b.evaluation+'</div>';"
"h+='</div></div>';});"
"document.getElementById('batchArea').innerHTML=h;}"
"function toggleBatch(idx){let bb=document.getElementById('bb'+idx);let bh=document.getElementById('bh'+idx);"
"if(bb.classList.contains('open')){bb.classList.remove('open');bh.textContent=t('expand');}"
"else{bb.classList.add('open');bh.textContent=t('collapse');}}"
"function pushInfer(d){"
"frameTimes.push(Date.now());if(frameTimes.length>100)frameTimes.shift();"
"document.getElementById('tot').textContent=d.total;"
"if(d.avgNpu)document.getElementById('npu').textContent=d.avgNpu.toFixed(2);"
"if(d.avgE2e)document.getElementById('e2e').textContent=d.avgE2e.toFixed(2);"
"document.getElementById('fps').textContent=calcFps();"
"if(document.getElementById('first').textContent==='--')document.getElementById('first').textContent=d.e2e.toFixed(2);"
"let lastEl=document.getElementById('last');"
"lastEl.textContent=d.class+' ('+d.conf.toFixed(3)+')';"
"lastEl.className='v'+(d.conf<LOW_TH?' lowconf':'');"
"rows.unshift(d);if(rows.length>30)rows.pop();"
"renderRows();updateModeTag();}"
"function pushBatch(d){batches.unshift(d);if(batches.length>10)batches.pop();renderBatches();}"
"fetch('/api/device/status').then(r=>r.json()).then(j=>{"
"curStatus='online';document.getElementById('dev').textContent=t('online');"
"document.getElementById('dev').className='v ok';"
"if(j.avgNpuMs)document.getElementById('npu').textContent=j.avgNpuMs;"
"document.getElementById('input').textContent=t('unknown');"
"}).catch(()=>{curStatus='reconn';});"
"let es=new EventSource('/api/stream');"
"es.onopen=()=>{curStatus='online';document.getElementById('dev').textContent=t('online');"
"document.getElementById('dev').className='v ok';};"
"es.onmessage=e=>{let d=JSON.parse(e.data);"
"if(d.type==='infer')pushInfer(d);else if(d.type==='batch')pushBatch(d);};"
"es.onerror=()=>{curStatus='reconn';document.getElementById('dev').textContent=t('reconn');"
"document.getElementById('dev').className='v bad';};"
"setInterval(()=>{document.getElementById('fps').textContent=calcFps();updateModeTag();},1000);"
"applyLang();"
"</script></body></html>";

static void send_json(int fd, const char* body){
    char hdr[128]; int n=snprintf(hdr,sizeof(hdr),
        "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n"
        "Access-Control-Allow-Origin: *\r\nContent-Length: %d\r\nConnection: close\r\n\r\n",(int)strlen(body));
    send(fd,hdr,n,0); send(fd,body,strlen(body),0);
}

static void* http_loop(void* arg){
    (void)arg;
    int lsn=socket(AF_INET,SOCK_STREAM,0); int opt=1;
    setsockopt(lsn,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
    struct sockaddr_in a; memset(&a,0,sizeof(a));
    a.sin_family=AF_INET; a.sin_addr.s_addr=htonl(INADDR_ANY); a.sin_port=htons(8081);
    if(bind(lsn,(struct sockaddr*)&a,sizeof(a))<0){perror("bind 8081");return NULL;}
    listen(lsn,8);
    printf("[INFO] Web(SSE) dashboard on :8081\n");
    char buf[1024];
    while(1){
        int cfd=accept(lsn,NULL,NULL); if(cfd<0)continue;
        int n=recv(cfd,buf,sizeof(buf)-1,0);
        if(n<=0){close(cfd);continue;}
        buf[n]=0;
        char path[128]={0};
        const char* sp=strchr(buf,' ');
        if(sp){ sscanf(sp+1,"%127s",path); }

        if(strcmp(path,"/api/stream")==0){
            const char* hdr="HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                            "Cache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
            send(cfd,hdr,strlen(hdr),0);
            const char* hello="event: open\ndata: {\"type\":\"hello\"}\n\n";
            send(cfd,hello,strlen(hello),0);
            sse_add(cfd);
            printf("[SSE] client connected (fd=%d)\n", cfd);
            continue;
        }
        if(strcmp(path,"/api/health")==0){
            const char* ok="HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                           "Content-Length: 2\r\nConnection: close\r\n\r\nok";
            send(cfd,ok,strlen(ok),0); close(cfd); continue;
        }

        pthread_mutex_lock(&G.m);
        double avgNpu = G.total>0 ? G.npuSum/G.total : 0;
        double avgE2e = G.total>0 ? G.e2eSum/G.total : 0;
        if(strcmp(path,"/api/device/status")==0){
            char j[1024];
            int pos=snprintf(j,sizeof(j),
              "{\"status\":\"online\",\"total\":%ld,\"avgNpuMs\":\"%.2f\",\"avgE2eMs\":\"%.2f\","
              "\"lastClass\":\"%s\",\"lastConf\":%.3f,\"uptimeSec\":%.0f,\"recent\":[",
              G.total, avgNpu, avgE2e, CLASS_NAMES[G.lastPred], G.lastConf,
              (now_ms()-G.bootMs)/1000.0);
            int first=1;
            for(int k=0;k<RECENT_N;k++){
                int idx=(G.ringHead-1-k+RECENT_N*2)%RECENT_N;
                if(G.ring[idx].npu==0) continue;
                pos+=snprintf(j+pos,sizeof(j)-pos,
                  "%s{\"class\":\"%s\",\"conf\":%.3f,\"npu\":%.2f,\"e2e\":%.2f}",
                  first?"":",", CLASS_NAMES[G.ring[idx].pred],
                  G.ring[idx].conf, G.ring[idx].npu, G.ring[idx].e2e);
                first=0;
            }
            snprintf(j+pos,sizeof(j)-pos,"]}");
            pthread_mutex_unlock(&G.m);
            send_json(cfd,j);
        } else {
            pthread_mutex_unlock(&G.m);
            char hdr[128];
            int hn=snprintf(hdr,sizeof(hdr),
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                "Content-Length: %d\r\nConnection: close\r\n\r\n",(int)strlen(HTML));
            send(cfd,hdr,hn,0);
            send(cfd,HTML,strlen(HTML),0);
        }
        close(cfd);
    }
    return NULL;
}

int main(int argc,char**argv){
    if(argc<2){printf("usage: %s model.rknn\n",argv[0]);return -1;}
    memset(&G,0,sizeof(G));
    G.bootMs=now_ms();
    pthread_mutex_init(&G.m,0);
    pthread_mutex_init(&batch_m,0);
    memset(&B,0,sizeof(B));
    sse_init();
    signal(SIGPIPE,SIG_IGN);
    if(rknn_setup(argv[1])<0) return -1;
    pthread_t t1,t2,t3;
    pthread_create(&t1,NULL,tcp_loop,NULL);
    pthread_create(&t2,NULL,http_loop,NULL);
    pthread_create(&t3,NULL,sse_heartbeat,NULL);
    printf("[INFO] ready. TCP=8080, Web(SSE)=http://<board-ip>:8081\n");
    pthread_join(t1,NULL);
    return 0;
}

