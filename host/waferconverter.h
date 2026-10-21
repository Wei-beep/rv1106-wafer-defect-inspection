#ifndef WAFERCONVERTER_H
#define WAFERCONVERTER_H

#include <QString>
#include <QByteArray>
#include <opencv2/opencv.hpp>

/**
 * 晶圆数据统一转换工具 (纯 C++, 无 Python 依赖)
 * 支持输入: .bin (4096B原始) / .png (彩色) / .stdf (SEMI v4)
 * 输出: 12352B = 64x64x3 uint8 RGB (12288B) + 16×float32 特征 (64B)
 * 特征提取复用 WaferFeature, 与推理链路完全一致
 */
class WaferConverter
{
public:
    static constexpr int EDGE_BIN_SIZE = 12352;
    static constexpr int IMG_BYTES = 12288;
    static constexpr int FEAT_BYTES = 64;

    /** 自动识别格式并转换, 返回 12352B edge bin (空表示失败) */
    static QByteArray convertAuto(const QString& path);

    /** 原始 BIN (64x64 uint8, 0/1/2, 4096B) → edge bin */
    static QByteArray convertRawBin(const QString& path);

    /** 彩色 PNG (export_png.py 格式) → edge bin */
    static QByteArray convertPng(const QString& path);

    /** STDF (SEMI v4, DRR 逐 die) → edge bin */
    static QByteArray convertStdf(const QString& path);

    /** 将 edge bin 保存到文件 */
    static bool saveEdgeBin(const QByteArray& data, const QString& path);

private:
    /** 64x64 uint8 晶圆图 (0/1/2) → 12352B edge bin */
    static QByteArray waferToEdgeBin(const cv::Mat& wafer);

    /** 从 STDF 字节流中解析晶圆图 */
    static cv::Mat parseStdf(const QByteArray& raw);
};

#endif // WAFERCONVERTER_H
