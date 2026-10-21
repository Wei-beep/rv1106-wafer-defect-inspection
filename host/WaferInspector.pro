QT       += core gui widgets network
TARGET = WaferInspector
TEMPLATE = app
SOURCES += \
    devicepanel.cpp \
    main.cpp \
    mainwindow.cpp \
    stdfparser.cpp \
    waferconverter.cpp \
    waferfeature.cpp \
    inferengine.cpp \
    waferconverter.cpp
HEADERS += \
    devicepanel.h \
    mainwindow.h \
    stdfparser.h \
    waferconverter.h \
    waferfeature.h \
    inferengine.h \
    waferconverter.h
# ========== OpenCV 4.12.0 配置 ==========
OPENCV_PATH = D:/OpenCV/opencv/build
INCLUDEPATH += $$OPENCV_PATH/include
CONFIG(debug, debug|release) {
    LIBS += -L$$OPENCV_PATH/x64/vc16/lib -lopencv_world4120d
} else {
    LIBS += -L$$OPENCV_PATH/x64/vc16/lib -lopencv_world4120
}
# ========== ONNX Runtime 1.16.3 配置 ==========
ONNXRUNTIME_PATH = D:/onnx/onnxruntime-win-x64-1.16.3
INCLUDEPATH += $$ONNXRUNTIME_PATH/include
LIBS += -L$$ONNXRUNTIME_PATH/lib -lonnxruntime
# 编译后自动复制运行时 dll
win32 {
   DEFINES += _CRT_SECURE_NO_WARNINGS
    # 强制MSVC按UTF-8解析源文件，彻底解决C2001编码错误
    QMAKE_CFLAGS += /utf-8
    QMAKE_CXXFLAGS += /utf-8
    CONFIG(debug, debug|release) {
        QMAKE_POST_LINK += $$quote(xcopy /Y "$$OPENCV_PATH/x64/vc16/bin/opencv_world4120d.dll" "$${OUT_PWD}" >nul 2>&1 & )
    } else {
        QMAKE_POST_LINK += $$quote(xcopy /Y "$$OPENCV_PATH/x64/vc16/bin/opencv_world4120.dll" "$${OUT_PWD}" >nul 2>&1 & )
    }
    QMAKE_POST_LINK += $$quote(xcopy /Y "$$ONNXRUNTIME_PATH/lib/onnxruntime.dll" "$${OUT_PWD}" >nul 2>&1 & )
}
FORMS += \
    mainwindow.ui
