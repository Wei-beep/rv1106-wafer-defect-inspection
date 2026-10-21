#include "waferconverter.h"
#include "waferfeature.h"
#include <QFile>
#include <QFileInfo>
#include <QDebug>
#include <cstring>

// ── 归一化参数 (与 mainwindow.cpp / 训练一致) ──
static const float FEAT_MEAN[16] = {
    0.5f, 0.5f, 0.2f, 0.05f, 0.02f,
    0,0,0,0,0,0,0,0,
    0.3f, 1.0f, 0.3f
};
static const float FEAT_STD[16] = {
    0.1f, 0.1f, 0.1f, 0.05f, 0.01f,
    0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,0.1f,
    0.1f, 0.3f, 0.1f
};

// ═══════════════════════════════════════════════════════════
//  公共: 晶圆图 → edge bin
// ═══════════════════════════════════════════════════════════
QByteArray WaferConverter::waferToEdgeBin(const cv::Mat& waferIn)
{
    cv::Mat wafer;
    if (waferIn.rows != 64 || waferIn.cols != 64) {
        cv::resize(waferIn, wafer, cv::Size(64, 64), 0, 0, cv::INTER_NEAREST);
    } else {
        wafer = waferIn;
    }

    // 图像: 0→(0,0,0), 1→(128,128,128), 2→(255,255,255), NHWC
    cv::Mat img(64, 64, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < 64; y++) {
        const uchar* wr = wafer.ptr<uchar>(y);
        cv::Vec3b* ir = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < 64; x++) {
            uchar v = wr[x];
            if (v == 1)      ir[x] = cv::Vec3b(128, 128, 128);
            else if (v == 2) ir[x] = cv::Vec3b(255, 255, 255);
        }
    }

    // 特征提取 (复用 WaferFeature)
    static WaferFeature wf;
    static bool normInit = false;
    if (!normInit) {
        wf.setNorm(FEAT_MEAN, FEAT_STD);
        normInit = true;
    }
    WaferFeature::FeatureVec feat = wf.extract(wafer, true);

    // 拼接: 12288B 图像 + 64B 特征 = 12352B
    QByteArray out;
    out.reserve(EDGE_BIN_SIZE);
    out.append(reinterpret_cast<const char*>(img.data), IMG_BYTES);
    out.append(reinterpret_cast<const char*>(feat.data()), FEAT_BYTES);
    return out;
}

// ═══════════════════════════════════════════════════════════
//  格式 1: 原始 BIN
// ═══════════════════════════════════════════════════════════
QByteArray WaferConverter::convertRawBin(const QString& path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return QByteArray();
    QByteArray data = f.readAll();
    f.close();

    cv::Mat wafer;
    if (data.size() == 4096) {
        // 64x64 uint8 原始
        wafer = cv::Mat(64, 64, CV_8UC1, const_cast<char*>(data.constData())).clone();
    } else if (data.size() == 12352) {
        // 已经是 edge bin, 反解晶圆图
        cv::Mat img(64, 64, CV_8UC3, const_cast<char*>(data.constData()));
        wafer = cv::Mat(64, 64, CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < 64; y++) {
            const cv::Vec3b* ir = img.ptr<cv::Vec3b>(y);
            uchar* wr = wafer.ptr<uchar>(y);
            for (int x = 0; x < 64; x++) {
                uchar v = ir[x][0];
                if (v >= 200)      wr[x] = 2;
                else if (v >= 64)  wr[x] = 1;
            }
        }
    } else {
        qWarning() << "WaferConverter: raw bin size" << data.size() << "not supported";
        return QByteArray();
    }
    return waferToEdgeBin(wafer);
}

// ═══════════════════════════════════════════════════════════
//  格式 2: PNG (彩色, export_png.py 格式)
// ═══════════════════════════════════════════════════════════
QByteArray WaferConverter::convertPng(const QString& path)
{
    cv::Mat img = cv::imread(path.toStdString(), cv::IMREAD_COLOR);  // BGR
    if (img.empty()) return QByteArray();

    int h = img.rows, w = img.cols;
    cv::Mat wafer(h, w, CV_8UC1, cv::Scalar(0));

    // 反向颜色映射 (BGR):
    //   失效 (0,0,255)  → B 通道高 → 2
    //   良品 (0,200,0)  → G 通道高 → 1
    //   背景 (30,30,30) → 其余     → 0
    for (int y = 0; y < h; y++) {
        const cv::Vec3b* ir = img.ptr<cv::Vec3b>(y);
        uchar* wr = wafer.ptr<uchar>(y);
        for (int x = 0; x < w; x++) {
            uchar b = ir[x][0], g = ir[x][1];
            if (b > 128)         wr[x] = 2;   // 失效
            else if (g > 128)    wr[x] = 1;   // 良品
            // else 背景
        }
    }
    return waferToEdgeBin(wafer);
}

// ═══════════════════════════════════════════════════════════
//  格式 3: STDF (SEMI v4)
// ═══════════════════════════════════════════════════════════
cv::Mat WaferConverter::parseStdf(const QByteArray& raw)
{
    const uchar* data = reinterpret_cast<const uchar*>(raw.constData());
    int total = raw.size();
    int pos = 0;
    int waferW = 64, waferH = 64;
    cv::Mat wafer;

    while (pos + 4 <= total) {
        uint16_t recLen = data[pos] | (data[pos+1] << 8);
        uchar recType = data[pos+2];
        uchar recSub = data[pos+3];
        pos += 4;
        const uchar* rd = data + pos;
        int rdLen = recLen;
        pos += recLen;

        // WIR (2,10): 晶圆信息, 含宽高
        if (recType == 2 && recSub == 10) {
            int rp = 7;  // skip <BBIB
            // wafer_id (cstr)
            if (rp < rdLen) { int sl = rd[rp]; rp += 1 + sl; }
            // defect_type (cstr)
            if (rp < rdLen) { int sl = rd[rp]; rp += 1 + sl; }
            // width, height (<HH)
            if (rp + 4 <= rdLen) {
                waferW = rd[rp] | (rd[rp+1] << 8);
                waferH = rd[rp+2] | (rd[rp+3] << 8);
            }
            wafer = cv::Mat(waferH, waferW, CV_8UC1, cv::Scalar(0));
        }
        // DRR (2,30): 逐 die 结果
        else if (recType == 2 && recSub == 30) {
            if (wafer.empty()) wafer = cv::Mat(waferH, waferW, CV_8UC1, cv::Scalar(0));
            // 格式: <BBIBhhB, x/y 在偏移 7, test_flag 在偏移 11
            if (rdLen >= 12) {
                int16_t x = rd[7] | (rd[8] << 8);
                int16_t y = rd[9] | (rd[10] << 8);
                uchar flag = rd[11];
                if (x >= 0 && x < waferW && y >= 0 && y < waferH) {
                    if (flag == 1)      wafer.at<uchar>(y, x) = 1;  // Pass
                    else if (flag == 3) wafer.at<uchar>(y, x) = 2;  // Fail
                }
            }
        }
    }
    return wafer;
}

QByteArray WaferConverter::convertStdf(const QString& path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return QByteArray();
    QByteArray raw = f.readAll();
    f.close();

    cv::Mat wafer = parseStdf(raw);
    if (wafer.empty()) return QByteArray();
    return waferToEdgeBin(wafer);
}

// ═══════════════════════════════════════════════════════════
//  自动识别 + 保存
// ═══════════════════════════════════════════════════════════
QByteArray WaferConverter::convertAuto(const QString& path)
{
    QString ext = QFileInfo(path).suffix().toLower();
    if (ext == "bin")  return convertRawBin(path);
    if (ext == "png")  return convertPng(path);
    if (ext == "stdf") return convertStdf(path);
    qWarning() << "WaferConverter: unsupported format" << ext;
    return QByteArray();
}

bool WaferConverter::saveEdgeBin(const QByteArray& data, const QString& path)
{
    if (data.size() != EDGE_BIN_SIZE) return false;
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly)) return false;
    f.write(data);
    f.close();
    return true;
}
