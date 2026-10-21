// =====================================================================
// RV1106 直解推理服务  wafer_direct.c
//   监控 incoming/ 目录, 吃到 .bin 立即 NPU 推理, SSE 推送到网页
//   BIN 格式(产线最优): 12288B (64x64x3 uint8 RGB NHWC) + 64B (16×float32 特征) = 12352B
//   HTTP/SSE 看板: 8081
//   用法: ./wafer_direct wafer_pc_int8.rknn [incoming_dir]
//         默认 incoming=./incoming, 处理后移到 ./processed
//   产线对接: 上位机/产线机台把 .bin 写入 incoming/ 即可, 板子自动消费
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
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include "rknn_api.h"

#define IMG_BYTES   (64*64*3)
#define FEAT_BYTES  (16*4)
#define BIN_BYTES   (IMG_BYTES+FEAT_BYTES)   /* 12352 */
#define NC 8
#define RECENT_N 32
#define SSE_MAX 8
#define POLL_MS   300
/* TCP 推理协议 (与 Qt edgeInfer 一致) */
#define REQ_BYTES   (4+IMG_BYTES+FEAT_BYTES)  /* "RFNS"+img+feat = 12356 */
#define RSP_BYTES   (4+4+4+NC*4)               /* "RFNR"+pred+conf+8logits = 44 */

static const char* CLASS_NAMES[NC] = {
    "Center","Donut","Edge-Loc","Edge-Ring","Loc","Near-full","Random","Scratch"
};
static const char* SUGGEST[NC] = {
    "光刻/对准","薄膜沉积/刻蚀","边缘曝光","边缘刻蚀/沉积",
    "局部工艺偏差","整片工艺异常","随机污染/颗粒","机械划伤/搬运"
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
    long   processedFiles;
    struct { int pred; float conf; double npu; double e2e; } ring[RECENT_N];
    int ringHead;
} Stats;
static Stats G;
static char g_incoming[256]="./incoming";
static char g_processed[256]="./processed";

/* 前向声明 (recordInfer 里调用) */
static void sse_broadcast(const char* data);

/* ---------------- 记录推理结果 (TCP模式调用) ---------------- */
static void recordInfer(int pred,float conf,double npu,double e2e,
                        int secondPred,float secondConf){
    pthread_mutex_lock(&G.m);
    G.total++;G.npuSum+=npu;G.e2eSum+=e2e;G.lastPred=pred;G.lastConf=conf;
    G.ring[G.ringHead].pred=pred;G.ring[G.ringHead].conf=conf;
    G.ring[G.ringHead].npu=npu;G.ring[G.ringHead].e2e=e2e;
    G.ringHead=(G.ringHead+1)%RECENT_N;
    double avgNpu=G.npuSum/G.total,avgE2e=G.e2eSum/G.total;
    long total=G.total;
    pthread_mutex_unlock(&G.m);
    char j[512];
    snprintf(j,sizeof(j),
      "{\"type\":\"infer\",\"class\":\"%s\",\"conf\":%.3f,\"npu\":%.2f,\"e2e\":%.2f,"
      "\"total\":%ld,\"avgNpu\":%.2f,\"avgE2e\":%.2f,"
      "\"secondClass\":\"%s\",\"secondConf\":%.3f,\"source\":\"TCP\"}",
      CLASS_NAMES[pred],conf,npu,e2e,total,avgNpu,avgE2e,
      CLASS_NAMES[secondPred],secondConf);
    sse_broadcast(j);
}

/* ---------------- SSE ---------------- */
static int sse_fds[SSE_MAX];
static pthread_mutex_t sse_m;
static void sse_init(void){pthread_mutex_init(&sse_m,0);for(int i=0;i<SSE_MAX;i++)sse_fds[i]=-1;}
static void sse_add(int fd){pthread_mutex_lock(&sse_m);for(int i=0;i<SSE_MAX;i++)if(sse_fds[i]<0){sse_fds[i]=fd;break;}pthread_mutex_unlock(&sse_m);}
static void sse_broadcast(const char* data){
    char buf[2048]; int n=snprintf(buf,sizeof(buf),"data: %s\n\n",data);
    pthread_mutex_lock(&sse_m);
    for(int i=0;i<SSE_MAX;i++){int fd=sse_fds[i];if(fd<0)continue;
        if(send(fd,buf,n,MSG_NOSIGNAL)<=0){close(fd);sse_fds[i]=-1;}}
    pthread_mutex_unlock(&sse_m);
}
static void* sse_heartbeat(void* arg){(void)arg;const char*hb=": heartbeat\n\n";
    while(1){sleep(10);pthread_mutex_lock(&sse_m);
        for(int i=0;i<SSE_MAX;i++){int fd=sse_fds[i];if(fd<0)continue;
            if(send(fd,hb,strlen(hb),MSG_NOSIGNAL)<=0){close(fd);sse_fds[i]=-1;}}
        pthread_mutex_unlock(&sse_m);}return NULL;}

/* ── UDP 广播: 每 2 秒向全网宣告板子存在, Qt 自动发现 ── */
#define DISCOVERY_PORT 8090
static void* udp_broadcast(void* arg){
    (void)arg;
    int fd=socket(AF_INET,SOCK_DGRAM,0);
    if(fd<0){printf("[WARN] udp socket failed\n");return NULL;}
    int bc=1;setsockopt(fd,SOL_SOCKET,SO_BROADCAST,&bc,sizeof(bc));
    struct sockaddr_in addr;memset(&addr,0,sizeof(addr));
    addr.sin_family=AF_INET;addr.sin_port=htons(DISCOVERY_PORT);
    addr.sin_addr.s_addr=htonl(INADDR_BROADCAST);
    /* 获取本机 IP */
    char myIp[32]="0.0.0.0";
    int tmp=socket(AF_INET,SOCK_DGRAM,0);
    if(tmp>=0){
        struct sockaddr_in srv;memset(&srv,0,sizeof(srv));
        srv.sin_family=AF_INET;srv.sin_port=htons(80);
        srv.sin_addr.s_addr=inet_addr("8.8.8.8");
        if(connect(tmp,(struct sockaddr*)&srv,sizeof(srv))==0){
            struct sockaddr_in local;socklen_t len=sizeof(local);
            getsockname(tmp,(struct sockaddr*)&local,&len);
            inet_ntop(AF_INET,&local.sin_addr,myIp,sizeof(myIp));
        }
        close(tmp);
    }
    char msg[256];
    while(1){
        int n=snprintf(msg,sizeof(msg),
            "{\"type\":\"wafer_direct\",\"ip\":\"%s\",\"http\":8081,\"tcp\":8080,\"model\":\"RV1106\"}",
            myIp);
        sendto(fd,msg,n,0,(struct sockaddr*)&addr,sizeof(addr));
        sleep(2);
    }
    return NULL;
}

static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static int recv_n(int fd,void*buf,int n){int r=0;while(r<n){int k=recv(fd,(char*)buf+r,n-r,0);if(k<=0)return r;r+=k;}return r;}
static int send_n(int fd,const void*buf,int n){int s=0;while(s<n){int k=send(fd,(const char*)buf+s,n-s,0);if(k<=0)return s;s+=k;}return s;}
static void* read_file(const char*p,int*n){FILE*f=fopen(p,"rb");if(!f)return NULL;fseek(f,0,SEEK_END);*n=ftell(f);fseek(f,0,SEEK_SET);void*b=malloc(*n);fread(b,1,*n,f);fclose(f);return b;}
static int8_t q1(float x,float s,int32_t z){float v=x/s+z;if(v>127)v=127;if(v<-128)v=-128;return(int8_t)(v+0.5f);}

/* ---------------- RKNN ---------------- */
static rknn_context ctx;
static rknn_tensor_attr ina[2], outa;
static rknn_tensor_mem *inm[2]={0},*outm=0;
static int ii,fi;
static int rknn_setup(const char* modelPath){
    int rc,mlen=0;void*model=read_file(modelPath,&mlen);
    if((rc=rknn_init(&ctx,model,mlen,0,NULL))<0){printf("rknn_init=%d\n",rc);return -1;}
    rknn_input_output_num ion;memset(&ion,0,sizeof(ion));rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&ion,sizeof(ion));
    for(uint32_t i=0;i<ion.n_input;i++){memset(&ina[i],0,sizeof(ina[i]));ina[i].index=i;
        rknn_query(ctx,RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR,&ina[i],sizeof(ina[i]));
        uint32_t al=ina[i].size_with_stride?ina[i].size_with_stride:ina[i].size;
        inm[i]=rknn_create_mem(ctx,al);rknn_set_io_mem(ctx,inm[i],&ina[i]);}
    memset(&outa,0,sizeof(outa));outa.index=0;rknn_query(ctx,RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR,&outa,sizeof(outa));
    outm=rknn_create_mem(ctx,outa.size_with_stride?outa.size_with_stride:outa.size);rknn_set_io_mem(ctx,outm,&outa);
    ii=ina[0].size>=IMG_BYTES?0:1;fi=1-ii;
    printf("[INFO] RKNN ready, in0=%s in1=%s out=%s\n",ina[0].name,ina[1].name,outa.name);return 0;
}
static int do_infer(const unsigned char* img,const float*feat,
                    int*pred,float*conf,float*logits,int*secondPred,float*secondConf){
    memcpy(inm[ii]->virt_addr,img,IMG_BYTES);
    int8_t*fd=(int8_t*)inm[fi]->virt_addr;
    for(int j=0;j<16;j++)fd[j]=q1(feat[j],ina[fi].scale,ina[fi].zp);
    if(rknn_run(ctx,NULL)<0)return -1;
    int8_t*oq=(int8_t*)outm->virt_addr;
    float mx=-1e9f,second=-1e9f;*pred=0;*secondPred=0;
    for(int k=0;k<NC;k++){logits[k]=(oq[k]-outa.zp)*outa.scale;
        if(logits[k]>mx){second=mx;*secondPred=*pred;mx=logits[k];*pred=k;}
        else if(logits[k]>second){second=logits[k];*secondPred=k;}}
    float se=0,tmp[NC];
    for(int k=0;k<NC;k++){tmp[k]=expf(logits[k]-mx);se+=tmp[k];}
    *conf=tmp[*pred]/se;*secondConf=tmp[*secondPred]/se;return 0;
}

/* ---------------- 直解: 目录监控 + 推理 ---------------- */
static void process_bin_file(const char* filepath, const char* filename){
    FILE*f=fopen(filepath,"rb");if(!f){printf("[ERR] cannot open %s\n",filepath);return;}
    static unsigned char buf[BIN_BYTES];
    int rd=fread(buf,1,BIN_BYTES,f);fclose(f);
    if(rd!=BIN_BYTES){printf("[WARN] %s size=%d, expected %d, skip\n",filename,rd,BIN_BYTES);return;}

    double t0=now_ms();
    int pred,secondPred;float conf,secondConf,lg[NC];
    if(do_infer(buf,(float*)(buf+IMG_BYTES),&pred,&conf,lg,&secondPred,&secondConf)<0){
        printf("[ERR] infer failed for %s\n",filename);return;
    }
    double npu=now_ms()-t0;
    double e2e=npu;  /* 直解模式: 无网络开销, e2e≈NPU推理 */

    pthread_mutex_lock(&G.m);
    G.total++;G.npuSum+=npu;G.e2eSum+=e2e;G.lastPred=pred;G.lastConf=conf;G.processedFiles++;
    G.ring[G.ringHead].pred=pred;G.ring[G.ringHead].conf=conf;
    G.ring[G.ringHead].npu=npu;G.ring[G.ringHead].e2e=e2e;
    G.ringHead=(G.ringHead+1)%RECENT_N;
    double avgNpu=G.npuSum/G.total,avgE2e=G.e2eSum/G.total;
    long total=G.total; long files=G.processedFiles;
    pthread_mutex_unlock(&G.m);

    /* 时间戳 */
    time_t raw=time(NULL);struct tm*ti=localtime(&raw);
    char ts[32];snprintf(ts,sizeof(ts),"%02d:%02d:%02d",ti->tm_hour,ti->tm_min,ti->tm_sec);

    /* SSE 推送 */
    char j[1024];
    snprintf(j,sizeof(j),
      "{\"type\":\"infer\",\"class\":\"%s\",\"conf\":%.3f,\"npu\":%.2f,\"e2e\":%.2f,"
      "\"total\":%ld,\"avgNpu\":%.2f,\"avgE2e\":%.2f,"
      "\"secondClass\":\"%s\",\"secondConf\":%.3f,"
      "\"filename\":\"%s\",\"timestamp\":\"%s\",\"source\":\"BIN直解\",\"files\":%ld}",
      CLASS_NAMES[pred],conf,npu,e2e,total,avgNpu,avgE2e,
      CLASS_NAMES[secondPred],secondConf,filename,ts,files);
    sse_broadcast(j);

    printf("[INFER] %s -> %s(%.3f) npu=%.2fms\n",filename,CLASS_NAMES[pred],conf,npu);

    /* 移到 processed/ */
    char dst[512];snprintf(dst,sizeof(dst),"%s/%s",g_processed,filename);
    rename(filepath,dst);
}

static void* direct_loop(void* arg){
    (void)arg;
    /* 确保目录存在 */
    mkdir(g_incoming,0755); mkdir(g_processed,0755);
    printf("[INFO] direct mode: watching %s/ (BIN %d bytes), output -> %s/\n",
           g_incoming,BIN_BYTES,g_processed);
    while(1){
        DIR*d=opendir(g_incoming);
        if(d){
            struct dirent*ent;
            while((ent=readdir(d))!=NULL){
                const char*name=ent->d_name;
                int len=strlen(name);
                if(len<5)continue;
                if(strcasecmp(name+len-4,".bin")!=0)continue;
                char filepath[512];snprintf(filepath,sizeof(filepath),"%s/%s",g_incoming,name);
                process_bin_file(filepath,name);
            }
            closedir(d);
        }
        usleep(POLL_MS*1000);
    }
    return NULL;
}

/* ---------------- HTTP + SSE 看板 ---------------- */
static const char* HTML =
"<!doctype html><html><head><meta charset=utf-8>"
"<meta name=viewport content='width=device-width,initial-scale=1'>"
"<title>Wafer Direct Monitor</title><style>"
"body{background:#0f1419;color:#e6e6e6;font-family:Segoe UI,Arial;margin:0;padding:20px}"
".topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}"
"h1{font-size:18px;margin:0;color:#4fc3f7}"
"#btnLang{background:#1a222b;color:#4fc3f7;border:1px solid #2a3540;border-radius:6px;padding:5px 12px;cursor:pointer;font-size:12px}"
".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:14px}"
".card{background:#1a222b;border:1px solid #2a3540;border-radius:8px;padding:12px}"
".v{font-size:22px;font-weight:700;color:#fff}.l{font-size:11px;color:#8aa0b3;margin-top:3px}"
"table{width:100%;border-collapse:collapse;margin-top:8px;font-size:12px}"
"td,th{padding:5px 7px;border-bottom:1px solid #26303a;text-align:left}"
".ok{color:#66bb6a}.bad{color:#ef5350}.lowconf{color:#ef5350;font-weight:700}"
".dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#66bb6a;margin-right:6px;animation:pulse 1.2s infinite}"
"@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}"
".tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px}"
".tag-run{background:#1565c0;color:#fff}.tag-idle{background:#424242;color:#bdbdbd}"
".sec{color:#8aa0b3}.fn{color:#4fc3f7;font-family:monospace;font-size:11px}"
"</style></head><body>"
"<div class='topbar'>"
"<h1><span class='dot'></span><span id='title'>RV1106 晶圆直解监控 (BIN 实时)</span></h1>"
"<button id='btnLang' onclick='toggleLang()'>EN</button>"
"</div>"
"<div class='grid'>"
"<div class='card'><div class='v' id='dev'>--</div><div class='l' id='lblStatus'>设备状态</div></div>"
"<div class='card'><div class='v' id='mode'><span class='tag tag-idle' id='modeTag'>Idle</span></div><div class='l' id='lblMode'>运行状态</div></div>"
"<div class='card'><div class='v' id='tot'>0</div><div class='l' id='lblTotal'>累计推理</div></div>"
"<div class='card'><div class='v' id='files'>0</div><div class='l' id='lblFiles'>已处理文件</div></div>"
"<div class='card'><div class='v' id='npu'>--</div><div class='l' id='lblNpu'>NPU 时延 (ms)</div></div>"
"<div class='card'><div class='v' id='fps'>--</div><div class='l' id='lblFps'>吞吐量 (帧/s)</div></div>"
"<div class='card'><div class='v' id='last' style='font-size:16px'>--</div><div class='l' id='lblLast'>最近判定</div></div>"
"<div class='card'><div class='v' id='input' style='font-size:14px'>BIN直解</div><div class='l' id='lblInput'>输入源</div></div>"
"</div>"
"<div class='card'><b id='lblStream'>实时推理流 (直解)</b>"
"<span style='float:right;font-size:11px;color:#8aa0b3' id='lblHint'>置信度 &lt; 0.50 标红</span>"
"<table><thead><tr>"
"<th>#</th><th id='thFile'>文件名</th><th id='thTime'>时间</th>"
"<th id='thClass'>类别</th><th id='thConf'>置信度</th>"
"<th id='thSec'>次选</th><th id='thNpu'>NPU ms</th><th id='thFlag'>标记</th>"
"</tr></thead><tbody id='tb'></tbody></table></div>"
"<script>"
"const LOW_TH=0.50;"
"const T={zh:{title:'RV1106 晶圆直解监控 (BIN 实时)',status:'设备状态',mode:'运行状态',"
"total:'累计推理',files:'已处理文件',npu:'NPU 时延 (ms)',fps:'吞吐量 (帧/s)',last:'最近判定',input:'输入源',"
"stream:'实时推理流 (直解)',hint:'置信度 < 0.50 标红',file:'文件名',time:'时间',cls:'类别',"
"conf:'置信度',sec:'次选',npuCol:'NPU ms',flag:'标记',low:'低置信',run:'运行中',idle:'空闲',"
"online:'在线',reconn:'重连中'},"
"en:{title:'RV1106 Wafer Direct Monitor (BIN Live)',status:'Status',mode:'Status',"
"total:'Total Inferences',files:'Files Processed',npu:'NPU Latency (ms)',fps:'Throughput (fps)',"
"last:'Last Result',input:'Source',stream:'Live Inference Stream (Direct)',hint:'Confidence < 0.50 highlighted',"
"file:'File',time:'Time',cls:'Class',conf:'Confidence',sec:'2nd',npuCol:'NPU ms',flag:'Flag',"
"low:'Low',run:'Running',idle:'Idle',online:'Online',reconn:'Reconnecting'}};"
"let lang=localStorage.getItem('wafer_lang')||'zh';let curStatus='offline';"
"function t(k){return(T[lang]&&T[lang][k])?T[lang][k]:k;}"
"function applyLang(){"
"document.getElementById('title').textContent=t('title');"
"document.getElementById('lblStatus').textContent=t('status');"
"document.getElementById('lblMode').textContent=t('mode');"
"document.getElementById('lblTotal').textContent=t('total');"
"document.getElementById('lblFiles').textContent=t('files');"
"document.getElementById('lblNpu').textContent=t('npu');"
"document.getElementById('lblFps').textContent=t('fps');"
"document.getElementById('lblLast').textContent=t('last');"
"document.getElementById('lblInput').textContent=t('input');"
"document.getElementById('lblStream').textContent=t('stream');"
"document.getElementById('lblHint').textContent=t('hint');"
"document.getElementById('thFile').textContent=t('file');"
"document.getElementById('thTime').textContent=t('time');"
"document.getElementById('thClass').textContent=t('cls');"
"document.getElementById('thConf').textContent=t('conf');"
"document.getElementById('thSec').textContent=t('sec');"
"document.getElementById('thNpu').textContent=t('npuCol');"
"document.getElementById('thFlag').textContent=t('flag');"
"document.getElementById('btnLang').textContent=lang==='zh'?'EN':'中';"
"document.getElementById('dev').textContent=t(curStatus);"
"document.getElementById('dev').className='v'+(curStatus==='online'?' ok':' bad');"
"renderRows();}"
"function toggleLang(){lang=lang==='zh'?'en':'zh';localStorage.setItem('wafer_lang',lang);applyLang();}"
"let rows=[];let frameTimes=[];"
"function updateMode(){"
"const now=Date.now();let recent=frameTimes.filter(ts=>now-ts<3000).length;"
"let tag=document.getElementById('modeTag');"
"if(recent>0){tag.textContent=t('run');tag.className='tag tag-run';}"
"else{tag.textContent=t('idle');tag.className='tag tag-idle';}}"
"function calcFps(){const now=Date.now();let r=frameTimes.filter(ts=>now-ts<5000).length;return r>0?(r/5.0).toFixed(1):'--';}"
"function renderRows(){"
"let h='';rows.forEach((q,i)=>{let low=q.conf<LOW_TH;"
"h+='<tr><td>'+(i+1)+'</td><td class=fn>'+q.filename+'</td><td>'+q.timestamp+'</td>';"
"h+='<td>'+q.class+'</td><td class='+(low?'lowconf':'')+'>'+q.conf.toFixed(3)+'</td>';"
"h+='<td class=sec>'+q.secondClass+'</td><td>'+q.npu.toFixed(2)+'</td>';"
"h+='<td>'+(low?'<span class=lowconf>'+t('low')+'</span>':'')+'</td></tr>';});"
"document.getElementById('tb').innerHTML=h;}"
"function pushInfer(d){"
"frameTimes.push(Date.now());if(frameTimes.length>100)frameTimes.shift();"
"document.getElementById('tot').textContent=d.total;"
"if(d.files)document.getElementById('files').textContent=d.files;"
"if(d.avgNpu)document.getElementById('npu').textContent=d.avgNpu.toFixed(2);"
"document.getElementById('fps').textContent=calcFps();"
"let lastEl=document.getElementById('last');"
"lastEl.textContent=d.class+' ('+d.conf.toFixed(3)+')';"
"lastEl.className='v'+(d.conf<LOW_TH?' lowconf':'');"
"rows.unshift(d);if(rows.length>50)rows.pop();renderRows();updateMode();}"
"fetch('/api/device/status').then(r=>r.json()).then(j=>{curStatus='online';"
"document.getElementById('dev').textContent=t('online');document.getElementById('dev').className='v ok';"
"if(j.avgNpuMs)document.getElementById('npu').textContent=j.avgNpuMs;"
"if(j.total)document.getElementById('tot').textContent=j.total;"
"if(j.files)document.getElementById('files').textContent=j.files;}).catch(()=>{curStatus='reconn';});"
"let es=new EventSource('/api/stream');"
"es.onopen=()=>{curStatus='online';document.getElementById('dev').textContent=t('online');document.getElementById('dev').className='v ok';};"
"es.onmessage=e=>{let d=JSON.parse(e.data);if(d.type==='infer')pushInfer(d);};"
"es.onerror=()=>{curStatus='reconn';document.getElementById('dev').textContent=t('reconn');document.getElementById('dev').className='v bad';};"
"setInterval(()=>{document.getElementById('fps').textContent=calcFps();updateMode();},1000);"
"applyLang();"
"</script></body></html>";

static void send_json(int fd,const char* body){
    char hdr[128];int n=snprintf(hdr,sizeof(hdr),
        "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n"
        "Content-Length: %d\r\nConnection: close\r\n\r\n",(int)strlen(body));
    send(fd,hdr,n,0);send(fd,body,strlen(body),0);
}

/* ── TCP 推理服务 (8080): 旧模式/研发模式, Qt edgeInfer 协议 ── */
static void* tcp_loop(void* arg){
    (void)arg;
    int lsn=socket(AF_INET,SOCK_STREAM,0);int opt=1;setsockopt(lsn,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
    struct sockaddr_in a;memset(&a,0,sizeof(a));a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_ANY);a.sin_port=htons(8080);
    if(bind(lsn,(struct sockaddr*)&a,sizeof(a))<0){perror("bind 8080");return NULL;}
    listen(lsn,4);printf("[INFO] TCP inference on :8080 (edge NPU mode)\n");
    static char req[REQ_BYTES];
    while(1){
        int cfd=accept(lsn,NULL,NULL);if(cfd<0)continue;
        while(1){
            char magic[4];
            if(recv_n(cfd,magic,4)!=4||memcmp(magic,"RFNS",4)!=0)break;
            if(recv_n(cfd,req,REQ_BYTES-4)!=REQ_BYTES-4)break;
            double tStart=now_ms();
            int pred,secondPred;float conf,secondConf,lg[NC];
            double t0=now_ms();
            if(do_infer((unsigned char*)req,(float*)(req+IMG_BYTES),
                         &pred,&conf,lg,&secondPred,&secondConf)<0)break;
            double npu=now_ms()-t0;
            char rsp[RSP_BYTES];char*rp=rsp;
            memcpy(rp,"RFNR",4);rp+=4;
            int32_t p32=pred;memcpy(rp,&p32,4);rp+=4;
            memcpy(rp,&conf,4);rp+=4;
            for(int k=0;k<NC;k++){memcpy(rp,&lg[k],4);rp+=4;}
            send_n(cfd,rsp,RSP_BYTES);
            double e2e=now_ms()-tStart;
            recordInfer(pred,conf,npu,e2e,secondPred,secondConf);
        }
        close(cfd);
    }
    return NULL;
}

static void* http_loop(void* arg){
    (void)arg;
    int lsn=socket(AF_INET,SOCK_STREAM,0);int opt=1;setsockopt(lsn,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
    struct sockaddr_in a;memset(&a,0,sizeof(a));a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_ANY);a.sin_port=htons(8081);
    if(bind(lsn,(struct sockaddr*)&a,sizeof(a))<0){perror("bind 8081");return NULL;}
    listen(lsn,8);printf("[INFO] Web(SSE) dashboard on :8081\n");
    char buf[4096];
    while(1){
        int cfd=accept(lsn,NULL,NULL);if(cfd<0)continue;
        int n=recv(cfd,buf,sizeof(buf)-1,0);if(n<=0){close(cfd);continue;}buf[n]=0;
        /* 解析方法和路径 */
        char method[8]={0},path[128]={0};
        sscanf(buf,"%7s %127s",method,path);
        int isPost=(strcmp(method,"POST")==0);

        /* ── POST /api/upload: 接收 edge bin, 存入 incoming/ ── */
        if(isPost && strncmp(path,"/api/upload",11)==0){
            /* 提取 Content-Length */
            int contentLen=0;
            char*cl=strstr(buf,"Content-Length:");
            if(cl)sscanf(cl,"Content-Length: %d",&contentLen);
            /* 提取文件名 (?name=xxx.bin) */
            char fname[128]="upload.bin";
            char*qn=strstr(path,"?name=");
            if(qn)snprintf(fname,sizeof(fname),"%s",qn+6);
            /* 找到 body 起始 (\r\n\r\n) */
            char*body=strstr(buf,"\r\n\r\n");
            int bodyOff=body?body-buf+4:n;
            int have=n-bodyOff;
            /* 保存文件 */
            char savePath[512];snprintf(savePath,sizeof(savePath),"%s/%s",g_incoming,fname);
            FILE*f=fopen(savePath,"wb");
            if(f){
                if(have>0)fwrite(buf+bodyOff,1,have,f);
                /* 读完剩余数据 */
                while(have<contentLen){
                    int rn=recv(cfd,buf,sizeof(buf),0);
                    if(rn<=0)break;
                    fwrite(buf,1,rn,f);have+=rn;
                }
                fclose(f);
            }
            char j[256];snprintf(j,sizeof(j),
                "{\"status\":\"ok\",\"filename\":\"%s\",\"bytes\":%d}",fname,have);
            send_json(cfd,j);
            printf("[UPLOAD] %s (%d bytes) -> incoming/\n",fname,have);
            close(cfd);continue;
        }

        if(strcmp(path,"/api/stream")==0){
            const char*h="HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
            send(cfd,h,strlen(h),0);
            const char*hello="event: open\ndata: {\"type\":\"hello\"}\n\n";send(cfd,hello,strlen(hello),0);
            sse_add(cfd);printf("[SSE] client connected\n");continue;
        }
        pthread_mutex_lock(&G.m);
        double avgNpu=G.total>0?G.npuSum/G.total:0;
        if(strcmp(path,"/api/device/status")==0){
            char j[512];snprintf(j,sizeof(j),
                "{\"status\":\"online\",\"total\":%ld,\"files\":%ld,\"avgNpuMs\":\"%.2f\"}",
                G.total,G.processedFiles,avgNpu);
            pthread_mutex_unlock(&G.m);send_json(cfd,j);
        }else{
            pthread_mutex_unlock(&G.m);
            char hdr[128];int hn=snprintf(hdr,sizeof(hdr),
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: %d\r\nConnection: close\r\n\r\n",(int)strlen(HTML));
            send(cfd,hdr,hn,0);send(cfd,HTML,strlen(HTML),0);
        }
        close(cfd);
    }
    return NULL;
}

int main(int argc,char**argv){
    if(argc<2){printf("usage: %s model.rknn [incoming_dir]\n",argv[0]);return -1;}
    if(argc>=3){snprintf(g_incoming,sizeof(g_incoming),"%s",argv[2]);}
    snprintf(g_processed,sizeof(g_processed),"%s/processed",g_incoming);
    /* 如果 incoming 是相对路径, processed 放在同级 */
    if(strcmp(g_incoming,"./incoming")==0){snprintf(g_processed,sizeof(g_processed),"./processed");}
    memset(&G,0,sizeof(G));G.bootMs=now_ms();
    pthread_mutex_init(&G.m,0);sse_init();signal(SIGPIPE,SIG_IGN);
    if(rknn_setup(argv[1])<0)return -1;
    pthread_t t1,t2,t3,t4,t5;
    pthread_create(&t1,NULL,direct_loop,NULL);
    pthread_create(&t2,NULL,http_loop,NULL);
    pthread_create(&t3,NULL,sse_heartbeat,NULL);
    pthread_create(&t4,NULL,udp_broadcast,NULL);
    pthread_create(&t5,NULL,tcp_loop,NULL);
    printf("[INFO] ready. TCP=:8080(edge NPU) + HTTP=:8081(direct+web) + UDP=:8090(discovery)\n");
    pthread_join(t1,NULL);
    return 0;
}


