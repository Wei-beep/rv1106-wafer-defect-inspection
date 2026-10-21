#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFile>
#include <QTextStream>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QColor>
#include <QShortcut>
#include <QProgressBar>
#include <QMenu>
#include <QDateTime>
#include <QBrush>
#include <QFont>
#include <QCloseEvent>
#include <QCoreApplication>
#include <QHBoxLayout>
#include <vector>
#include <algorithm>
#include "stdfparser.h"
#include "waferfeature.h"   // 特征提取，边缘推理要用

const float FEAT_MEAN[] = {
    0.5f, 0.5f, 0.2f, 0.05f,
    0.02f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.3f, 1.0f, 0.3f
};
const float FEAT_STD[] = {
    0.1f, 0.1f, 0.1f, 0.05f,
    0.01f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    0.1f, 0.3f, 0.1f
};

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    // 强制读取exe同级配置
    QString configPath = QDir(QCoreApplication::applicationDirPath()).filePath("config.ini");
    m_config = new QSettings(configPath, QSettings::IniFormat, this);
    loadConfig();
    setWindowTitle("[操作员] 晶圆缺陷模式识别系统");
    resize(1200, 720);
    setMinimumSize(1000, 650);
    // 全局工业风样式
    setStyleSheet(R"(
        QPushButton {
            padding: 6px 14px;
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            background: #f8f8f8;
        }
        QPushButton:hover { background: #e8f4ff; border-color: #4a90e2; }
        QPushButton:disabled { color: #aaa; background: #f0f0f0; }
        QComboBox { padding: 4px 10px; border: 1px solid #d0d0d0;
                    border-radius: 4px; background: #fff; }
        QTableWidget {
            gridline-color: #e5e5e5;
            selection-background-color: #d6e8ff;
        }
        QHeaderView::section {
            padding: 6px;
            background: #f0f0f0;
            border: none;
            border-right: 1px solid #e0e0e0;
            font-weight: bold;
        }
    )");
    // 布局拉伸
    ui->grid_main->setRowStretch(0, 0);
    ui->grid_main->setRowStretch(1, 1);
    ui->grid_main->setColumnStretch(0, 5);
    ui->grid_main->setColumnStretch(1, 5);
    ui->grid_right->setRowStretch(0, 0);
    ui->grid_right->setRowStretch(1, 0);
    ui->grid_right->setRowStretch(2, 6);
    ui->grid_right->setRowStretch(3, 0);
    ui->grid_right->setRowStretch(4, 3);
    // 状态栏进度条
    m_progressBar = new QProgressBar(this);
    m_progressBar->setFixedWidth(220);
    m_progressBar->setVisible(false);
    statusBar()->addPermanentWidget(m_progressBar);
    // 全局快捷键
    m_shortcutPrev = new QShortcut(QKeySequence("A"), this);
    m_shortcutNext = new QShortcut(QKeySequence("D"), this);
    connect(m_shortcutPrev, &QShortcut::activated, this, &MainWindow::slot_prevWafer);
    connect(m_shortcutNext, &QShortcut::activated, this, &MainWindow::slot_nextWafer);
    qApp->installEventFilter(this);
    // 表格右键菜单
    ui->tableResult->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(ui->tableResult, &QTableWidget::customContextMenuRequested,
            this, [this](const QPoint& pos){
                QMenu menu(this);
                QAction *actExp = menu.addAction(m_isZh ? "导出选中" : "Export Selected");
                QAction *actDel = menu.addAction(m_isZh ? "删除选中行" : "Delete Selected");
                actDel->setEnabled(m_currentRole == ROLE_ADMIN);
                QAction *res = menu.exec(ui->tableResult->viewport()->mapToGlobal(pos));
                if (res == actExp) slot_exportCsv();
                else if (res == actDel) slot_deleteSelected();
            });
    // 信号槽
    connect(ui->btnSingleOpen, &QPushButton::clicked, this, &MainWindow::slot_openSingleFile);
    connect(ui->btnBatch, &QPushButton::clicked, this, &MainWindow::slot_batchInfer);
    connect(ui->btnExport, &QPushButton::clicked, this, &MainWindow::slot_exportCsv);
    connect(ui->btnGlobalHeat, &QPushButton::clicked, this, &MainWindow::slot_toggleGlobalHeat);
    connect(ui->btnRole, &QPushButton::clicked, this, &MainWindow::slot_switchRole);
    connect(ui->cbShowHeatSingle, &QCheckBox::toggled, this, &MainWindow::slot_toggleSingleHeat);
    connect(ui->cbTTA, &QCheckBox::toggled, this, &MainWindow::slot_ttaToggled);
    connect(ui->comboLang, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::slot_switchLang);
    connect(ui->tableResult, &QTableWidget::cellClicked,
            this, &MainWindow::slot_tableRowClicked);
    connect(ui->tableResult, &QTableWidget::cellDoubleClicked,
            this, &MainWindow::slot_tableRowDoubleClicked);
    connect(ui->btnPrev, &QPushButton::clicked, this, &MainWindow::slot_prevWafer);
    connect(ui->btnNext, &QPushButton::clicked, this, &MainWindow::slot_nextWafer);
    connect(ui->comboFilterType, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::slot_filterChanged);
    connect(ui->comboFilterClass, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::slot_filterChanged);
    connect(ui->cbFilterLow, &QCheckBox::toggled,
            this, &MainWindow::slot_filterChanged);
    ui->btnPrev->setEnabled(false);
    ui->btnNext->setEnabled(false);

    // ========== 新增：构建标准菜单栏 ==========
    buildMenuBar();
    // 动作按钮收敛进菜单栏，顶栏仅保留状态开关
    ui->btnSingleOpen->hide();
    ui->btnBatch->hide();
    ui->btnExport->hide();
    ui->btnGlobalHeat->hide();

    // ========== 新增：推理后端切换 + 设备连接按钮 ==========
    QWidget *toolbarWidget = new QWidget(this);
    QHBoxLayout *toolbarLayout = new QHBoxLayout(toolbarWidget);
    toolbarLayout->setContentsMargins(0, 0, 0, 0);
    m_backendLabel = new QLabel(m_isZh ? "推理后端：" : "Backend:");
    toolbarLayout->addWidget(m_backendLabel);

    m_comboBackend = new QComboBox(this);
    m_comboBackend->addItem(m_isZh ? "本地CPU" : "Local CPU");
    m_comboBackend->addItem(m_isZh ? "RV1106边缘NPU" : "RV1106 Edge NPU");
    connect(m_comboBackend, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::slot_switchBackend);
    toolbarLayout->addWidget(m_comboBackend);

    m_deviceBtn = new QPushButton(m_isZh ? "设备连接" : "Device", this);
    connect(m_deviceBtn, &QPushButton::clicked, this, &MainWindow::slot_openDeviceDialog);
    toolbarLayout->addWidget(m_deviceBtn);

    // 打开网页看板按钮
    m_webBtn = new QPushButton(m_isZh ? "网页看板" : "Web Dashboard", this);
    connect(m_webBtn, &QPushButton::clicked, this, &MainWindow::slot_openWebDashboard);
    toolbarLayout->addWidget(m_webBtn);

    // 导出并下发到板端直解
    m_exportBtn = new QPushButton(m_isZh ? "导出并下发" : "Export & Upload", this);
    connect(m_exportBtn, &QPushButton::clicked, this, &MainWindow::slot_exportAndUpload);
    toolbarLayout->addWidget(m_exportBtn);

    // 设备自动发现状态标签
    m_discoveryLabel = new QLabel(m_isZh ? "未发现设备" : "No device", this);
    m_discoveryLabel->setStyleSheet("color:#888;font-size:11px;padding:0 8px;");
    toolbarLayout->addWidget(m_discoveryLabel);

    toolbarLayout->addStretch();

    // 放入底部状态栏永久区，避免遮挡顶栏的角色/语言按钮
    statusBar()->addPermanentWidget(toolbarWidget);

    // ========== UDP 自动发现板子 (监听 8090) ==========
    m_udpSocket = new QUdpSocket(this);
    m_udpSocket->bind(QHostAddress::Any, 8090, QUdpSocket::ShareAddress);
    connect(m_udpSocket, &QUdpSocket::readyRead, this, &MainWindow::slot_udpDiscovery);

    // ========== 新增：设备连接面板实例 ==========
    m_devicePanel = new DevicePanel(this);

    // 加载模型
    if (!m_engine.load(m_modelPath.toStdString(), FEAT_MEAN, FEAT_STD)) {
        QString err = m_isZh ? "模型加载失败，请检查配置文件路径" : "Model load failed, check config path";
        QMessageBox::critical(this, m_isZh?"错误":"Error", err);
        appendAuditLog("模型加载失败", m_modelPath);
    } else {
        appendLog("[信息] 模型加载成功，推理引擎就绪  |  [INFO] Model loaded, engine ready");
    }
    checkBreakpointTask();
    applyRolePermission();
    refreshUiText();

    // ========== 启动模式选择 (每次都弹, 不记住) ==========
    buildProductionPanel();
    // 默认隐藏单图热力 (复选框未勾选时不应显示)
    ui->labelHeatSingle->setVisible(false);
    // 如果配置里有监控文件夹, 自动启动监控
    if (!m_watchFolder.isEmpty() && QDir(m_watchFolder).exists()) {
        m_folderWatcher->addPath(m_watchFolder);
        m_watchTimer->start();
        if (m_prodWatchLabel) {
            m_prodWatchLabel->setText(m_isZh ? QString("监控中: %1").arg(m_watchFolder)
                                             : QString("Watching: %1").arg(m_watchFolder));
            m_prodWatchLabel->setStyleSheet("font-size:12px;color:#4caf50;");
        }
    }
    m_currentMode = showModeSelectDialog();
    applyMode(m_currentMode);

    appendAuditLog("程序启动");
}

MainWindow::~MainWindow()
{
    saveConfig();
    delete ui;
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    appendAuditLog("程序退出");
    saveConfig();
    event->accept();
}

// ==================== 配置与权限 ====================
void MainWindow::loadConfig()
{
    m_modelPath = m_config->value("System/modelPath", "D:/AIProject/paper/output/wafer_model.onnx").toString();
    m_exportDir = m_config->value("System/exportDir", "./").toString();
    m_lowConfThreshold = m_config->value("System/lowConfThreshold", DEFAULT_CONF_THRESHOLD).toFloat();
    m_normalConfThreshold = m_config->value("System/normalThreshold", DEFAULT_NORMAL_THRESHOLD).toFloat();
    m_useTTA = m_config->value("System/ttaDefault", false).toBool();
    m_currentRole = ROLE_OPERATOR;   // 启动固定操作员模式，避免默认管理员
    m_adminPassword = m_config->value("System/adminPassword", "admin").toString();
    m_breakpointFolder = m_config->value("Breakpoint/folder", "").toString();
    m_breakpointProcessed = m_config->value("Breakpoint/processed", 0).toInt();
    m_hasBreakpoint = !m_breakpointFolder.isEmpty();
    m_watchFolder = m_config->value("System/watchFolder", "").toString();
}

void MainWindow::saveConfig()
{
    m_config->setValue("System/ttaDefault", m_useTTA);
    m_config->setValue("System/lastRole", m_currentRole);
    m_config->setValue("System/watchFolder", m_watchFolder);
    m_config->sync();
}

void MainWindow::applyRolePermission()
{
    bool isAdmin = (m_currentRole == ROLE_ADMIN);
    // 操作员开放全部常用功能
    ui->cbTTA->setEnabled(true);
    ui->comboFilterType->setEnabled(true);
    ui->comboFilterClass->setEnabled(true);
    ui->cbFilterLow->setEnabled(true);
    ui->cbShowHeatSingle->setEnabled(true);
    ui->btnGlobalHeat->setEnabled(true);
    ui->btnExport->setEnabled(true);
    ui->btnRole->setEnabled(true);
    QString roleTag = isAdmin ? (m_isZh ? "[管理员]" : "[Admin]")
                              : (m_isZh ? "[操作员]" : "[Operator]");
    QString titleBase = m_isZh ? "基于边缘AI的晶圆缺陷检测上位机软件 V1.0" : "EdgeAI Wafer Defect Inspection Host Software V1.0";
    setWindowTitle(QString("%1 %2").arg(roleTag).arg(titleBase));
}

void MainWindow::slot_switchRole()
{
    bool ok;
    QString pwd = QInputDialog::getText(this,
                                        m_isZh ? "身份验证" : "Authentication",
                                        m_isZh ? "请输入管理员密码：" : "Enter admin password:",
                                        QLineEdit::Password, "", &ok);
    if (!ok) return;
    if (pwd != m_adminPassword) {
        QMessageBox::warning(this, m_isZh?"错误":"Error",
                             m_isZh ? "密码错误" : "Wrong password");
        return;
    }
    m_currentRole = ROLE_ADMIN;
    applyRolePermission();
    appendAuditLog("切换角色", "管理员");
}

// ==================== 审计日志 ====================
void MainWindow::appendAuditLog(const QString& action, const QString& detail)
{
    QFile file("audit.log");
    if (file.open(QIODevice::Append | QIODevice::Text)) {
        QTextStream out(&file);
        out.setCodec("UTF-8");
        QString time = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
        QString role = (m_currentRole == ROLE_ADMIN) ? "管理员" : "操作员";
        out << QString("[%1] [%2] %3 %4\n").arg(time).arg(role).arg(action).arg(detail);
        file.close();
    }
}

// ==================== 断点续传 ====================
bool MainWindow::checkBreakpointTask()
{
    if (!m_hasBreakpoint) return false;
    int ret = QMessageBox::question(this,
                                    m_isZh ? "未完成任务" : "Unfinished Task",
                                    m_isZh ? QString("检测到上次未完成的批量任务：\n%1\n是否继续处理？").arg(m_breakpointFolder)
                                           : QString("Found unfinished batch task:\n%1\nResume processing?").arg(m_breakpointFolder),
                                    QMessageBox::Yes | QMessageBox::No);
    if (ret != QMessageBox::Yes) {
        clearBreakpoint();
        return false;
    }
    QString folder = m_breakpointFolder;
    int startIdx = m_breakpointProcessed;
    QDir dir(folder);
    QStringList filters = {"*.bin", "*.png", "*.stdf"};
    QFileInfoList fileList = dir.entryInfoList(filters, QDir::Files);
    int total = fileList.size();
    if (startIdx >= total) {
        clearBreakpoint();
        return false;
    }
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, total);
    m_progressBar->setValue(startIdx);
    QDateTime batchTime = QDateTime::currentDateTime();
    int resultStart = (int)m_allResults.size();
    for (int i = startIdx; i < total; i++) {
        const QFileInfo& fi = fileList[i];
        cv::Mat wafer = loadWaferFile(fi.absoluteFilePath());
        if (wafer.empty()) continue;

        int top1Id, top2Id;
        float top1Conf, top2Conf;
        std::array<float, 8> probs;

        if (m_useEdgeNpu) {
            bool ok = edgeInfer(wafer, top1Id, top1Conf, probs);
            if (!ok) {
                appendLog("[错误] 边缘推理失败，跳过当前文件");
                continue;
            }
        } else {
            InferEngine::Result res = m_engine.infer(wafer, m_useTTA);
            top1Id = res.classId;
            top1Conf = res.confidence;
            probs = res.probs;
        }

        getTop2(probs, top1Id, top1Conf, top2Id, top2Conf);
        int td, pd, fd;
        float y, d, cx, cy, cp, ar;
        calcWaferStats(wafer, td, pd, fd, y, d, cx, cy, cp, ar);
        InferenceResult r;
        r.fileName = fi.fileName();
        r.fullPath = fi.absoluteFilePath();
        r.importTime = batchTime;
        r.sourceType = "batch";
        r.classId = top1Id;
        r.className = QString::fromStdString(InferEngine::classNames()[top1Id]);
        r.confidence = top1Conf;
        r.secondClassId = top2Id;
        r.secondClassName = QString::fromStdString(InferEngine::classNames()[top2Id]);
        r.secondConfidence = top2Conf;
        r.isLowConfidence = top1Conf < m_lowConfThreshold;
        r.isNormal = false;
        r.totalDie = td; r.passDie = pd; r.failDie = fd;
        r.yieldRate = y; r.defectDensity = d;
        r.cx = cx; r.cy = cy;
        r.compactness = cp; r.aspectRatio = ar;
        r.probs = probs;
        postProcessResult(r, wafer);
        m_allResults.push_back(r);
        m_progressBar->setValue(i + 1);
        saveBreakpoint(folder, i + 1);
        qApp->processEvents();
    }
    m_progressBar->setVisible(false);
    clearBreakpoint();
    appendAuditLog("批量续传完成", folder);
    buildBatchGroups();
    rebuildFilter();
    buildDisplayRows();
    fillResultTable();
    if (!m_allResults.empty()) {
        m_currentIndex = resultStart;
        showWaferByIndex(m_currentIndex);
        ui->btnPrev->setEnabled(m_currentIndex > 0);
        ui->btnNext->setEnabled(m_currentIndex < (int)m_allResults.size() - 1);
    }
    return true;
}

void MainWindow::saveBreakpoint(const QString& folder, int processed)
{
    m_config->setValue("Breakpoint/folder", folder);
    m_config->setValue("Breakpoint/processed", processed);
    m_config->sync();
}

void MainWindow::clearBreakpoint()
{
    m_config->remove("Breakpoint/folder");
    m_config->remove("Breakpoint/processed");
    m_config->sync();
    m_hasBreakpoint = false;
    m_breakpointFolder.clear();
    m_breakpointProcessed = 0;
}

// ==================== 事件过滤器 ====================
bool MainWindow::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() == QEvent::Wheel && !m_showGlobalHeat && !m_allResults.empty()) {
        QWheelEvent *wheel = static_cast<QWheelEvent*>(event);
        QPoint globalPos(wheel->globalX(), wheel->globalY());
        QRect waferRect = ui->labelWaferView->rect();
        waferRect.moveTopLeft(ui->labelWaferView->mapToGlobal(QPoint(0, 0)));
        QRect heatRect = ui->labelHeatSingle->rect();
        heatRect.moveTopLeft(ui->labelHeatSingle->mapToGlobal(QPoint(0, 0)));
        QRect detailRect = ui->labelDetail->rect();
        detailRect.moveTopLeft(ui->labelDetail->mapToGlobal(QPoint(0, 0)));
        QRect leftRect = waferRect.united(heatRect).united(detailRect);
        if (leftRect.contains(globalPos)) {
            if (wheel->angleDelta().y() > 0) slot_prevWafer();
            else slot_nextWafer();
            return true;
        }
    }
    return QMainWindow::eventFilter(obj, event);
}

// ==================== 后处理规则（工程化修正） ====================
void MainWindow::postProcessResult(InferenceResult& r, const cv::Mat& wafer)
{
    // 规则1：正常判定
    if (r.confidence < m_normalConfThreshold) {
        r.classId = -1;
        r.className = m_isZh ? "正常" : "Normal";
        r.isNormal = true;
        r.isLowConfidence = false;
        return;
    }
    // 规则2：Scratch不可能出现在中心
    if (r.className == "Scratch" && r.cx > 0.3 && r.cx < 0.7 && r.cy > 0.3 && r.cy < 0.7) {
        r.className = "Loc";
        r.classId = 4;
        r.confidence *= 0.8f;
        r.isLowConfidence = r.confidence < m_lowConfThreshold;
    }
    // 规则3：Near-full不可能只有边缘少量失效
    if (r.className == "Near-full" && r.defectDensity < 0.4f) {
        r.className = "Random";
        r.classId = 5;
        r.confidence *= 0.75f;
        r.isLowConfidence = r.confidence < m_lowConfThreshold;
    }
}

// ==================== 热力图 ====================
void MainWindow::buildGlobalHeatMap()
{
    if (m_allResults.empty()) {
        m_globalHeatMap.release();
        return;
    }
    cv::Mat countMat = cv::Mat::zeros(64, 64, CV_32SC1);
    for (const auto& r : m_allResults) {
        if (r.isNormal) continue;
        cv::Mat wafer = loadWaferFile(r.fullPath);
        if (wafer.empty()) continue;
        for (int y = 0; y < 64; y++) {
            const uchar* row = wafer.ptr<uchar>(y);
            int* cntRow = countMat.ptr<int>(y);
            for (int x = 0; x < 64; x++) {
                if (row[x] == 2) cntRow[x]++;
            }
        }
    }
    cv::Mat floatMat;
    countMat.convertTo(floatMat, CV_32F);
    floatMat += 1.0f;
    cv::log(floatMat, floatMat);
    std::vector<float> vals;
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++)
            if (countMat.at<int>(y,x) > 0)
                vals.push_back(floatMat.at<float>(y,x));
    std::sort(vals.begin(), vals.end());
    float maxVal = vals.empty() ? 1.0f : vals[(int)(vals.size()*0.90)];
    if (maxVal < 1e-6) maxVal = 1.0f;
    cv::Mat normMat;
    floatMat.convertTo(normMat, CV_8UC1, 255.0 / maxVal);
    cv::threshold(normMat, normMat, 255, 255, cv::THRESH_TRUNC);
    cv::applyColorMap(normMat, m_globalHeatMap, cv::COLORMAP_JET);
    for (int y = 0; y < 64; y++) {
        uchar* dstRow = m_globalHeatMap.ptr<uchar>(y);
        const int* cntRow = countMat.ptr<int>(y);
        for (int x = 0; x < 64; x++) {
            if (cntRow[x] == 0) {
                dstRow[x*3] = 30; dstRow[x*3+1] = 30; dstRow[x*3+2] = 30;
            }
        }
    }
}

QPixmap MainWindow::heatMapToPixmap(const cv::Mat& heatMap)
{
    QImage img(heatMap.data, heatMap.cols, heatMap.rows, (int)heatMap.step, QImage::Format_RGB888);
    return QPixmap::fromImage(img.rgbSwapped());
}

QPixmap MainWindow::singleWaferHeatMap(const cv::Mat& wafer)
{
    cv::Mat defectMask = cv::Mat::zeros(64, 64, CV_8UC1);
    for (int y = 0; y < 64; y++) {
        const uchar* row = wafer.ptr<uchar>(y);
        uchar* dst = defectMask.ptr<uchar>(y);
        for (int x = 0; x < 64; x++) {
            if (row[x] == 2) dst[x] = 255;
        }
    }
    cv::Mat blurMat;
    cv::GaussianBlur(defectMask, blurMat, cv::Size(7, 7), 1.2);
    cv::Mat heatColor;
    cv::applyColorMap(blurMat, heatColor, cv::COLORMAP_JET);
    for (int y = 0; y < 64; y++) {
        const uchar* src = defectMask.ptr<uchar>(y);
        uchar* dst = heatColor.ptr<uchar>(y);
        for (int x = 0; x < 64; x++) {
            if (src[x] == 0 && blurMat.at<uchar>(y,x) < 25) {
                dst[x*3] = 30; dst[x*3+1] = 30; dst[x*3+2] = 30;
            }
        }
    }
    QImage img(heatColor.data, heatColor.cols, heatColor.rows, (int)heatColor.step, QImage::Format_RGB888);
    return QPixmap::fromImage(img.rgbSwapped());
}

void MainWindow::slot_toggleGlobalHeat()
{
    m_showGlobalHeat = !m_showGlobalHeat;
    if (m_showGlobalHeat) {
        buildGlobalHeatMap();
        if (m_globalHeatMap.empty()) {
            QMessageBox::information(this, m_isZh?"提示":"Hint",
                                     m_isZh ? "暂无数据生成热力图" : "No data for heatmap");
            m_showGlobalHeat = false;
            return;
        }
        QPixmap heatPix = heatMapToPixmap(m_globalHeatMap);
        ui->labelWaferView->setPixmap(heatPix.scaled(
            ui->labelWaferView->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        ui->labelHeatSingle->setVisible(false);
        ui->btnGlobalHeat->setText(m_isZh ? "显示原图" : "Show Wafer");
        ui->labelResult->setText(m_isZh ? "【全局失效热力图】\n所有样本失效叠加密度" : "[Global Heatmap]\nAll samples defect density");
        ui->labelDetail->setText(m_isZh
                                     ? "统计范围：全部已导入晶圆\n颜色越深表示该位置失效次数越多\n蓝色：低频 → 红色：高频\n工程意义：定位批次共性失效区域，辅助工艺排查"
                                     : "Stats: all imported wafers\nDarker = higher failure frequency\nBlue: low → Red: high\nEngineering: identify common failure regions");
    } else {
        if (m_currentIndex >= 0 && m_currentIndex < (int)m_allResults.size()) {
            showWaferByIndex(m_currentIndex);
        } else {
            ui->labelWaferView->clear();
            ui->labelResult->setText(m_isZh ? "等待推理..." : "Waiting...");
            ui->labelDetail->clear();
        }
        ui->btnGlobalHeat->setText(m_isZh ? "全局热力图" : "Global Heatmap");
        ui->labelHeatSingle->setVisible(m_showSingleHeat);
    }
    appendAuditLog("切换全局热力图", m_showGlobalHeat ? "开启" : "关闭");
}

void MainWindow::slot_toggleSingleHeat(bool checked)
{
    m_showSingleHeat = checked;
    ui->labelHeatSingle->setVisible(checked);
    if (checked && m_currentIndex >= 0 && !m_showGlobalHeat) {
        cv::Mat wafer = loadWaferFile(m_allResults[m_currentIndex].fullPath);
        if (!wafer.empty()) {
            ui->labelHeatSingle->setPixmap(singleWaferHeatMap(wafer).scaled(
                ui->labelHeatSingle->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
    }
}

// ==================== 基础槽函数 ====================
void MainWindow::slot_ttaToggled(bool checked)
{
    m_useTTA = checked;
    appendLog(m_isZh ? QString("TTA测试时增强：%1").arg(checked?"开启":"关闭")
                     : QString("TTA Augmentation: %1").arg(checked?"ON":"OFF"));
    appendAuditLog("切换TTA", checked ? "开启" : "关闭");
}

void MainWindow::slot_switchLang(int idx)
{
    m_isZh = (idx == 0);
    refreshUiText();
    if (m_showGlobalHeat) {
        ui->btnGlobalHeat->setText(m_isZh ? "显示原图" : "Show Wafer");
    } else {
        ui->btnGlobalHeat->setText(m_isZh ? "全局热力图" : "Global Heatmap");
    }
}

// ==================== 单文件推理 ====================
void MainWindow::slot_openSingleFile()
{
    QString filter = m_isZh ? "晶圆文件 (*.bin *.png *.stdf)" : "Wafer files (*.bin *.png *.stdf)";
    QString filePath = QFileDialog::getOpenFileName(this, m_isZh?"选择晶圆文件":"Select file", "", filter);
    if (filePath.isEmpty()) return;
    appendLog(m_isZh
                  ? QString("开始单文件推理：%1").arg(QFileInfo(filePath).fileName())
                  : QString("Start single inference: %1").arg(QFileInfo(filePath).fileName()));
    cv::Mat wafer = loadWaferFile(filePath);
    if (wafer.empty()) {
        QMessageBox::warning(this, m_isZh?"错误":"Error", m_isZh?"文件读取失败":"Read failed");
        return;
    }

    int top1Id, top2Id;
    float top1Conf, top2Conf;
    std::array<float, 8> probs;

    if (m_useEdgeNpu) {
        bool ok = edgeInfer(wafer, top1Id, top1Conf, probs);
        if (!ok) {
            QMessageBox::warning(this, "错误", "边缘推理失败，请检查设备连接");
            return;
        }
    } else {
        InferEngine::Result result = m_engine.infer(wafer, m_useTTA);
        top1Id = result.classId;
        top1Conf = result.confidence;
        probs = result.probs;
    }

    getTop2(probs, top1Id, top1Conf, top2Id, top2Conf);
    int total, pass, fail;
    float yield, dens, cx, cy, compact, ar;
    calcWaferStats(wafer, total, pass, fail, yield, dens, cx, cy, compact, ar);
    InferenceResult r;
    r.fileName = QFileInfo(filePath).fileName();
    r.fullPath = filePath;
    r.importTime = QDateTime::currentDateTime();
    r.sourceType = "single";
    r.classId = top1Id;
    r.className = QString::fromStdString(InferEngine::classNames()[top1Id]);
    r.confidence = top1Conf;
    r.secondClassId = top2Id;
    r.secondClassName = QString::fromStdString(InferEngine::classNames()[top2Id]);
    r.secondConfidence = top2Conf;
    r.isLowConfidence = top1Conf < m_lowConfThreshold;
    r.isNormal = false;
    r.totalDie = total; r.passDie = pass; r.failDie = fail;
    r.yieldRate = yield; r.defectDensity = dens;
    r.cx = cx; r.cy = cy;
    r.compactness = compact; r.aspectRatio = ar;
    r.probs = probs;
    postProcessResult(r, wafer);
    m_allResults.push_back(r);
    m_currentIndex = (int)m_allResults.size() - 1;
    if (m_showGlobalHeat) {
        m_showGlobalHeat = false;
        ui->btnGlobalHeat->setText(m_isZh ? "全局热力图" : "Global Heatmap");
    }
    buildBatchGroups();
    rebuildFilter();
    buildDisplayRows();
    fillResultTable();
    showWaferByIndex(m_currentIndex);
    ui->btnPrev->setEnabled(m_currentIndex > 0);
    ui->btnNext->setEnabled(false);
    appendAuditLog("单文件推理", r.fileName);
    appendLog(m_isZh
                  ? QString("结果：%1 -> %2  置信度 %3")
                        .arg(r.fileName, r.className).arg(r.confidence, 0, 'f', 4)
                  : QString("Result: %1 -> %2  conf %3")
                        .arg(r.fileName, r.className).arg(r.confidence, 0, 'f', 4));
}

// ==================== 批量推理 ====================
void MainWindow::slot_batchInfer()
{
    QString folder = QFileDialog::getExistingDirectory(this, m_isZh?"选择批量文件夹":"Select folder", "");
    if (folder.isEmpty()) return;
    QDir dir(folder);
    QStringList filters = {"*.bin", "*.png", "*.stdf"};
    QFileInfoList fileList = dir.entryInfoList(filters, QDir::Files);
    int total = fileList.size();
    if (total == 0) return;
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, total);
    m_progressBar->setValue(0);
    appendLog(m_isZh ? QString("开始批量推理，共 %1 个文件").arg(total)
                     : QString("Start batch inference, %1 files").arg(total));
    appendAuditLog("开始批量推理", folder);
    QDateTime batchTime = QDateTime::currentDateTime();
    int startIdx = (int)m_allResults.size();
    for (int i = 0; i < total; i++) {
        const QFileInfo& fi = fileList[i];
        cv::Mat wafer = loadWaferFile(fi.absoluteFilePath());
        if (wafer.empty()) continue;

        int top1Id, top2Id;
        float top1Conf, top2Conf;
        std::array<float, 8> probs;

        if (m_useEdgeNpu) {
            bool ok = edgeInfer(wafer, top1Id, top1Conf, probs);
            if (!ok) {
                appendLog("[错误] 边缘推理失败，跳过当前文件");
                continue;
            }
        } else {
            InferEngine::Result res = m_engine.infer(wafer, m_useTTA);
            top1Id = res.classId;
            top1Conf = res.confidence;
            probs = res.probs;
        }

        getTop2(probs, top1Id, top1Conf, top2Id, top2Conf);
        int td, pd, fd;
        float y, d, cx, cy, cp, ar;
        calcWaferStats(wafer, td, pd, fd, y, d, cx, cy, cp, ar);
        InferenceResult r;
        r.fileName = fi.fileName();
        r.fullPath = fi.absoluteFilePath();
        r.importTime = batchTime;
        r.sourceType = "batch";
        r.classId = top1Id;
        r.className = QString::fromStdString(InferEngine::classNames()[top1Id]);
        r.confidence = top1Conf;
        r.secondClassId = top2Id;
        r.secondClassName = QString::fromStdString(InferEngine::classNames()[top2Id]);
        r.secondConfidence = top2Conf;
        r.isLowConfidence = top1Conf < m_lowConfThreshold;
        r.isNormal = false;
        r.totalDie = td; r.passDie = pd; r.failDie = fd;
        r.yieldRate = y; r.defectDensity = d;
        r.cx = cx; r.cy = cy;
        r.compactness = cp; r.aspectRatio = ar;
        r.probs = probs;
        postProcessResult(r, wafer);
        m_allResults.push_back(r);
        m_progressBar->setValue(i + 1);
        saveBreakpoint(folder, i + 1);
        qApp->processEvents();
    }
    m_progressBar->setVisible(false);
    clearBreakpoint();
    appendLog(m_isZh ? QString("批量完成，共处理 %1 个文件").arg(total)
                     : QString("Batch done, %1 files processed").arg(total));
    appendAuditLog("批量完成", QString("%1 个文件").arg(total));

    // ===== 批次整体评价 =====
    {
        int cnt[8] = {0};
        int normalCnt = 0;
        int n = (int)m_allResults.size() - startIdx;
        for (int k = startIdx; k < (int)m_allResults.size(); k++) {
            cnt[m_allResults[k].classId]++;
            if (m_allResults[k].isNormal) normalCnt++;
        }
        const char* procZh[8] = {
            "光刻/对准工艺", "薄膜沉积/刻蚀",
            "边缘曝光", "边缘刻蚀/划片",
            "局部污染", "整片工艺异常",
            "综合排查", "机械划伤/传片"
        };
        const char* procEn[8] = {
            "litho/alignment", "deposition/etch",
            "edge exposure", "edge etching/dicing",
            "local contamination", "wafer-wide anomaly",
            "general check", "scratch/handling"
        };
        int dom = 0;
        for (int k = 1; k < 8; k++) if (cnt[k] > cnt[dom]) dom = k;
        double domRatio = n > 0 ? (double)cnt[dom] / n : 0.0;

        int batchNo = m_batchReports.size() + 1;
        QStringList rep;
        rep << "========================================";
        rep << (m_isZh
                    ? QString("批次 #%1   时间: %2").arg(batchNo).arg(batchTime.toString("yyyy-MM-dd HH:mm:ss"))
                    : QString("Batch #%1   Time: %2").arg(batchNo).arg(batchTime.toString("yyyy-MM-dd HH:mm:ss")));
        rep << (m_isZh
                    ? QString("本批共 %1 片").arg(n)
                    : QString("Total: %1 wafers").arg(n));
        for (int k = 0; k < 8; k++) {
            if (cnt[k] == 0) continue;
            rep << QString("  %1  %2 %3  (%4.1f%%)")
                       .arg(QString::fromStdString(InferEngine::classNames()[k]), -12)
                       .arg(cnt[k])
                       .arg(m_isZh ? "片" : "pcs")
                       .arg(100.0 * cnt[k] / n, 0, 'f', 1);
        }
        rep << (m_isZh
                    ? QString("正常片: %1 (%2.1f%%)").arg(normalCnt).arg(100.0 * normalCnt / n)
                    : QString("Normal: %1 (%2.1f%%)").arg(normalCnt).arg(100.0 * normalCnt / n));
        rep << (m_isZh
                    ? QString("主导缺陷: %1 (%2.1f%%) -> 建议排查 %3")
                          .arg(QString::fromStdString(InferEngine::classNames()[dom]))
                          .arg(100.0 * domRatio).arg(QString::fromUtf8(procZh[dom]))
                    : QString("Dominant defect: %1 (%2.1f%%) -> check %3")
                          .arg(QString::fromStdString(InferEngine::classNames()[dom]))
                          .arg(100.0 * domRatio).arg(QString::fromUtf8(procEn[dom])));
        if (domRatio > 0.5)
            rep << (m_isZh ? "整体评价: 本批缺陷集中，建议停线排查工艺"
                           : "Overall: defects concentrated, recommend stopping to inspect process");
        else if (normalCnt > n * 0.8)
            rep << (m_isZh ? "整体评价: 本批良品率较高，整体良好"
                           : "Overall: high yield, batch looks good");
        else
            rep << (m_isZh ? "整体评价: 缺陷分散，建议抽检复核"
                           : "Overall: defects scattered, suggest sampling review");
        rep << "========================================";
        for (const QString& line : rep) appendLog(line);
        m_batchReports << rep.join("\n");
    }
    if (m_showGlobalHeat) {
        m_showGlobalHeat = false;
        ui->btnGlobalHeat->setText(m_isZh ? "全局热力图" : "Global Heatmap");
    }
    buildBatchGroups();
    rebuildFilter();
    buildDisplayRows();
    fillResultTable();
    if (!m_allResults.empty()) {
        m_currentIndex = startIdx;
        showWaferByIndex(m_currentIndex);
        ui->btnPrev->setEnabled(m_currentIndex > 0);
        ui->btnNext->setEnabled(m_currentIndex < (int)m_allResults.size() - 1);
    }
}

// ==================== CSV导出 ====================
void MainWindow::slot_exportCsv()
{
    if (m_allResults.empty()) {
        QMessageBox::information(this, m_isZh?"提示":"Hint", m_isZh?"暂无数据可导出":"No data to export");
        return;
    }
    QSet<int> selectedGlobal;
    QModelIndexList selected = ui->tableResult->selectionModel()->selectedRows();
    for (const auto& idx : selected) {
        int row = idx.row();
        if (row < 0 || row >= m_displayRows.size()) continue;
        if (m_displayRows[row].first) {
            QString key = m_batchOrder[m_displayRows[row].second];
            for (int gIdx : m_batchGroups[key]) {
                selectedGlobal.insert(gIdx);
            }
        } else {
            selectedGlobal.insert(m_displayRows[row].second);
        }
    }
    enum ExportMode { All, Selected };
    ExportMode mode = All;
    if (!selectedGlobal.isEmpty()) {
        int ret = QMessageBox::question(this, m_isZh?"导出选择":"Export Options",
                                        m_isZh ? "检测到选中项，是否仅导出选中内容？\n\n是：导出选中\n否：导出全部"
                                               : "Selected items detected.\nExport selected only?\n\nYes: export selected\nNo: export all",
                                        QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
        if (ret == QMessageBox::Cancel) return;
        mode = (ret == QMessageBox::Yes) ? Selected : All;
    }
    QVector<int> exportIndices;
    if (mode == Selected) {
        for (int idx : selectedGlobal) exportIndices.push_back(idx);
    } else {
        for (int i = 0; i < (int)m_allResults.size(); i++) exportIndices.push_back(i);
    }
    if (exportIndices.isEmpty()) {
        QMessageBox::information(this, m_isZh?"提示":"Hint", m_isZh?"没有可导出的数据":"No data to export");
        return;
    }
    QString defaultName = QString("%1/wafer_result_%2.csv")
                              .arg(m_exportDir)
                              .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
    QString savePath = QFileDialog::getSaveFileName(this,
                                                    m_isZh?"导出CSV":"Export CSV", defaultName, "CSV files (*.csv)");
    if (savePath.isEmpty()) return;
    if (QFile::exists(savePath)) {
        int ret = QMessageBox::question(this, m_isZh?"确认":"Confirm",
                                        m_isZh ? "文件已存在，是否覆盖？" : "File exists. Overwrite?");
        if (ret != QMessageBox::Yes) return;
    }
    QFile file(savePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, m_isZh?"错误":"Error", m_isZh?"文件写入失败":"Write failed");
        return;
    }
    // 先直接写入 UTF-8 BOM 三字节，保证 Excel 识别
    {
        const unsigned char bom[3] = {0xEF, 0xBB, 0xBF};
        file.write((const char*)bom, 3);
    }
    // 直接用 fromUtf8 构造 QString，再写 UTF-8 字节，避免 QTextStream 二次编码
    QString csv;
    csv += QString::fromUtf8(m_isZh
                                 ? "序号,文件名,导入时间,类型,预测类别,置信度,次选类别,次选置信度,备注,"
                                   "总Die数,良品数,失效数,良率(%),失效密度(%),重心X,重心Y,紧凑度,长宽比\n"
                                 : "Index,FileName,Import Time,Type,Class,Confidence,2nd Class,2nd Conf.,Remark,"
                                   "Total Die,Pass Die,Fail Die,Yield(%),Defect Density(%),Center X,Center Y,Compactness,Aspect Ratio\n");
    for (int i = 0; i < (int)exportIndices.size(); i++) {
        const auto& r = m_allResults[exportIndices[i]];
        QString type = r.sourceType == "batch"
                           ? QString::fromUtf8(m_isZh ? "批量" : "Batch")
                           : QString::fromUtf8(m_isZh ? "单次" : "Single");
        QString remark = r.isLowConfidence
                             ? QString::fromUtf8(m_isZh ? "置信度偏低" : "Low confidence")
                             : "";
        if (r.isNormal) remark = QString::fromUtf8(m_isZh ? "正常" : "Normal");
        csv += QString("%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18\n")
                   .arg(i+1)
                   .arg(r.fileName)
                   .arg(r.importTime.toString("yyyy-MM-dd HH:mm:ss"))
                   .arg(type)
                   .arg(r.className)
                   .arg(r.confidence, 0, 'f', 6)
                   .arg(r.secondClassName)
                   .arg(r.secondConfidence, 0, 'f', 6)
                   .arg(remark)
                   .arg(r.totalDie)
                   .arg(r.passDie)
                   .arg(r.failDie)
                   .arg(r.yieldRate*100, 0, 'f', 2)
                   .arg(r.defectDensity*100, 0, 'f', 2)
                   .arg(r.cx, 0, 'f', 4)
                   .arg(r.cy, 0, 'f', 4)
                   .arg(r.compactness, 0, 'f', 4)
                   .arg(r.aspectRatio, 0, 'f', 4);
    }
    for (int b = 0; b < m_batchReports.size(); b++) {
        csv += "\n" + m_batchReports[b] + "\n";
    }
    file.write(csv.toUtf8());
    file.close();
    QMessageBox::information(this,
                             m_isZh ? "导出成功" : "Export Success",
                             m_isZh ? QString("已导出 %1 条记录\n文件路径：%2").arg(exportIndices.size()).arg(savePath)
                                    : QString("Exported %1 records\nPath: %2").arg(exportIndices.size()).arg(savePath));
    appendAuditLog("导出CSV", QString("%1条记录").arg(exportIndices.size()));
}

// ==================== 删除选中行 ====================
void MainWindow::slot_deleteSelected()
{
    if (m_currentRole != ROLE_ADMIN) {
        QMessageBox::information(this, m_isZh?"权限不足":"Permission Denied",
                                 m_isZh ? "仅管理员可删除数据" : "Only admin can delete data");
        return;
    }
    QModelIndexList selected = ui->tableResult->selectionModel()->selectedRows();
    if (selected.isEmpty()) return;
    QVector<int> toDelete;
    for (const auto& idx : selected) {
        int row = idx.row();
        if (row >= 0 && row < m_displayRows.size() && !m_displayRows[row].first) {
            toDelete.push_back(m_displayRows[row].second);
        }
    }
    std::sort(toDelete.begin(), toDelete.end(), std::greater<int>());
    for (int gIdx : toDelete) {
        if (gIdx == m_currentIndex) {
            m_currentIndex = -1;
            ui->labelWaferView->clear();
            ui->labelHeatSingle->clear();
            ui->labelResult->setText(m_isZh ? "等待推理..." : "Waiting...");
            ui->labelDetail->clear();
        } else if (gIdx < m_currentIndex) {
            m_currentIndex--;
        }
        m_allResults.erase(m_allResults.begin() + gIdx);
    }
    buildBatchGroups();
    rebuildFilter();
    buildDisplayRows();
    fillResultTable();
    updateNavLabel();
    appendAuditLog("删除记录", QString("%1条").arg(toDelete.size()));
}

// ==================== 表格交互 ====================
void MainWindow::slot_tableRowClicked(int row, int)
{
    if (row < 0 || row >= (int)m_displayRows.size()) return;
    if (m_displayRows[row].first) return;
    if (m_showGlobalHeat) {
        m_showGlobalHeat = false;
        ui->btnGlobalHeat->setText(m_isZh ? "全局热力图" : "Global Heatmap");
    }
    m_currentIndex = m_displayRows[row].second;
    showWaferByIndex(m_currentIndex);
    ui->btnPrev->setEnabled(m_currentIndex > 0);
    ui->btnNext->setEnabled(m_currentIndex < (int)m_allResults.size() - 1);
}

void MainWindow::slot_tableRowDoubleClicked(int row, int)
{
    if (row < 0 || row >= (int)m_displayRows.size()) return;
    if (!m_displayRows[row].first) return;
    QString batchKey = m_batchOrder[m_displayRows[row].second];
    if (m_expandedBatches.contains(batchKey)) {
        m_expandedBatches.remove(batchKey);
    } else {
        m_expandedBatches.insert(batchKey);
    }
    buildDisplayRows();
    fillResultTable();
}

// ==================== 导航切换 ====================
void MainWindow::slot_prevWafer()
{
    if (m_showGlobalHeat) return;
    if (m_currentIndex > 0) {
        m_currentIndex--;
        showWaferByIndex(m_currentIndex);
        ui->btnNext->setEnabled(true);
        ui->btnPrev->setEnabled(m_currentIndex > 0);
        for (int i = 0; i < (int)m_displayRows.size(); i++) {
            if (!m_displayRows[i].first && m_displayRows[i].second == m_currentIndex) {
                ui->tableResult->selectRow(i);
                break;
            }
        }
    }
}

void MainWindow::slot_nextWafer()
{
    if (m_showGlobalHeat) return;
    if (m_currentIndex < (int)m_allResults.size() - 1) {
        m_currentIndex++;
        showWaferByIndex(m_currentIndex);
        ui->btnPrev->setEnabled(true);
        ui->btnNext->setEnabled(m_currentIndex < (int)m_allResults.size() - 1);
        for (int i = 0; i < (int)m_displayRows.size(); i++) {
            if (!m_displayRows[i].first && m_displayRows[i].second == m_currentIndex) {
                ui->tableResult->selectRow(i);
                break;
            }
        }
    }
}

// ==================== 筛选 ====================
void MainWindow::slot_filterChanged()
{
    int typeIdx = ui->comboFilterType->currentIndex();
    if (typeIdx == 0) m_filterType = "All";
    else if (typeIdx == 1) m_filterType = "single";
    else m_filterType = "batch";
    int classIdx = ui->comboFilterClass->currentIndex();
    if (classIdx == 0) m_filterClass = "All";
    else m_filterClass = ui->comboFilterClass->currentText();
    m_filterOnlyLow = ui->cbFilterLow->isChecked();
    rebuildFilter();
    buildDisplayRows();
    fillResultTable();
}

// ==================== 分组与筛选 ====================
void MainWindow::buildBatchGroups()
{
    m_batchGroups.clear();
    m_batchOrder.clear();
    for (int i = 0; i < (int)m_allResults.size(); i++) {
        if (m_allResults[i].sourceType != "batch") continue;
        QString key = m_allResults[i].importTime.toString("yyyy-MM-dd HH:mm:ss");
        m_batchGroups[key].push_back(i);
        if (!m_batchOrder.contains(key)) {
            m_batchOrder.prepend(key);
        }
    }
}

void MainWindow::rebuildFilter()
{
    m_filteredData.clear();
    for (int i = 0; i < (int)m_allResults.size(); i++) {
        const auto& r = m_allResults[i];
        if (m_filterOnlyLow && !r.isLowConfidence) continue;
        if (m_filterType != "All" && r.sourceType != m_filterType) continue;
        if (m_filterClass != "All" && r.className != m_filterClass) continue;
        m_filteredData.push_back(i);
    }
}

void MainWindow::buildDisplayRows()
{
    m_displayRows.clear();
    QSet<int> filteredSet;
    for (int idx : m_filteredData) filteredSet.insert(idx);
    for (int i = 0; i < (int)m_allResults.size(); i++) {
        if (m_allResults[i].sourceType == "single" && filteredSet.contains(i)) {
            m_displayRows.append(qMakePair(false, i));
        }
    }
    for (int bi = 0; bi < m_batchOrder.size(); bi++) {
        const QString& key = m_batchOrder[bi];
        const auto& indices = m_batchGroups[key];
        bool hasVisible = false;
        for (int idx : indices) {
            if (filteredSet.contains(idx)) { hasVisible = true; break; }
        }
        if (!hasVisible) continue;
        m_displayRows.append(qMakePair(true, bi));
        if (m_expandedBatches.contains(key)) {
            for (int idx : indices) {
                if (filteredSet.contains(idx)) {
                    m_displayRows.append(qMakePair(false, idx));
                }
            }
        }
    }
}

// ==================== 表格填充 ====================
void MainWindow::fillResultTable()
{
    ui->tableResult->clearSpans();
    int totalRows = (int)m_displayRows.size();
    int dataCount = (int)m_filteredData.size();
    int lowCount = 0;
    float avgConf = 0.0f;
    for (int idx : m_filteredData) {
        avgConf += m_allResults[idx].confidence;
        if (m_allResults[idx].isLowConfidence) lowCount++;
    }
    avgConf = dataCount > 0 ? avgConf / dataCount : 0.0f;
    if (m_isZh) {
        ui->labelResultTitle->setText(QString("识别结果（共%1片 | 平均置信度%2 | 异常%3片）")
                                          .arg(dataCount).arg(avgConf, 0, 'f', 3).arg(lowCount));
    } else {
        ui->labelResultTitle->setText(QString("Results (Total: %1 | Avg Conf: %2 | Alerts: %3)")
                                          .arg(dataCount).arg(avgConf, 0, 'f', 3).arg(lowCount));
    }
    ui->tableResult->setRowCount(totalRows);
    ui->tableResult->setColumnCount(6);
    QStringList headers;
    if (m_isZh) {
        headers << "文件名" << "预测类别" << "置信度" << "次选类别" << "次选置信度" << "备注";
    } else {
        headers << "File" << "Class" << "Confidence" << "2nd Class" << "2nd Conf." << "Remark";
    }
    ui->tableResult->setHorizontalHeaderLabels(headers);
    QFont groupFont;
    groupFont.setBold(true);
    QBrush groupBg(QColor(240, 240, 240));
    for (int row = 0; row < totalRows; ++row) {
        bool isGroup = m_displayRows[row].first;
        if (isGroup) {
            int bi = m_displayRows[row].second;
            QString key = m_batchOrder[bi];
            int count = m_batchGroups[key].size();
            bool expanded = m_expandedBatches.contains(key);
            QString text = m_isZh
                               ? QString("%1  [批量] %2  (%3片)  %4")
                                     .arg(expanded ? "▼" : "▶")
                                     .arg(key).arg(count)
                                     .arg(expanded ? "点击折叠" : "双击展开")
                               : QString("%1  [Batch] %2  (%3 pcs)  %4")
                                     .arg(expanded ? "▼" : "▶")
                                     .arg(key).arg(count)
                                     .arg(expanded ? "Collapse" : "Expand");
            QTableWidgetItem *item = new QTableWidgetItem(text);
            item->setFont(groupFont);
            item->setBackground(groupBg);
            item->setFlags(item->flags() & ~Qt::ItemIsSelectable);
            ui->tableResult->setItem(row, 0, item);
            ui->tableResult->setSpan(row, 0, 1, 6);
        } else {
            int idx = m_displayRows[row].second;
            const auto& r = m_allResults[idx];
            ui->tableResult->setItem(row, 0, new QTableWidgetItem(r.fileName));
            ui->tableResult->setItem(row, 1, new QTableWidgetItem(r.className));
            ui->tableResult->setItem(row, 2, new QTableWidgetItem(QString::number(r.confidence, 'f', 4)));
            ui->tableResult->setItem(row, 3, new QTableWidgetItem(r.secondClassName));
            ui->tableResult->setItem(row, 4, new QTableWidgetItem(QString::number(r.secondConfidence, 'f', 4)));
            QString remark;
            if (r.isNormal) remark = m_isZh ? "正常" : "Normal";
            else if (r.isLowConfidence) remark = m_isZh ? "置信度偏低" : "Low conf.";
            ui->tableResult->setItem(row, 5, new QTableWidgetItem(remark));
            if (r.isLowConfidence && !r.isNormal) {
                for (int col = 0; col < 6; ++col) {
                    ui->tableResult->item(row, col)->setBackground(QColor(255, 245, 200));
                }
            }
        }
    }
    ui->tableResult->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui->tableResult->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui->tableResult->setSelectionBehavior(QAbstractItemView::SelectRows);
    ui->tableResult->setAlternatingRowColors(false);
}

// ==================== 工具函数 ====================
void MainWindow::getTop2(const std::array<float, 8>& probs,
                         int& top1Id, float& top1Conf,
                         int& top2Id, float& top2Conf)
{
    top1Id = 0; top2Id = 0;
    top1Conf = probs[0]; top2Conf = 0.0f;
    for (int i = 1; i < 8; ++i) {
        if (probs[i] > top1Conf) {
            top2Id = top1Id; top2Conf = top1Conf;
            top1Id = i; top1Conf = probs[i];
        } else if (probs[i] > top2Conf) {
            top2Id = i; top2Conf = probs[i];
        }
    }
}

QString MainWindow::getProcessSuggestion(int classId)
{
    switch (classId) {
    case 0: return m_isZh
                   ? "• 匀胶转速不均，中心胶厚异常\n• 光刻中心聚焦偏差\n• 刻蚀中心区域气体分布不均"
                   : "• Spin coating speed mismatch\n• Photolithography center offset\n• Etching gas distribution anomaly";
    case 1: return m_isZh
                   ? "• 薄膜沉积环形厚度不均\n• 匀胶回吸参数异常\n• 中心区域颗粒污染"
                   : "• Film deposition ring mismatch\n• Spin coat back pressure anomaly\n• Center particle contamination";
    case 2: return m_isZh
                   ? "• 晶圆边缘局部颗粒掉落\n• 边缘吸盘接触损伤\n• 刻蚀边缘局部气流异常"
                   : "• Local particle drop on edge\n• Edge chuck contact damage\n• Local edge gas flow anomaly";
    case 3: return m_isZh
                   ? "• 刻蚀边缘效应（负载效应）\n• 薄膜沉积边缘厚度衰减\n• 边缘光刻胶回流"
                   : "• Etching edge load effect\n• Film edge thickness decay\n• Edge photoresist reflow";
    case 4: return m_isZh
                   ? "• 局部颗粒污染\n• 针点式缺陷/微掩模\n• 局部光刻散焦"
                   : "• Local particle contamination\n• Pinhole / micro mask\n• Local defocus in lithography";
    case 5: return m_isZh
                   ? "• 随机颗粒污染（环境/气体）\n• 工艺腔体杂质脱落\n• 传输过程随机落尘"
                   : "• Random particle contamination\n• Chamber debris peeling\n• Random dust during transfer";
    case 6: return m_isZh
                   ? "• 机械手传输划痕\n• 晶圆盒接触摩擦\n• 探针卡/吸盘划伤"
                   : "• Robot transfer scratch\n• Cassette contact friction\n• Probe card scratch";
    case 7: return m_isZh
                   ? "• 整面工艺失效\n• 气体供应中断\n• 电源/射频异常终止"
                   : "• Full wafer process failure\n• Gas supply interruption\n• Power / RF abort";
    default: return m_isZh ? "• 晶圆无明显缺陷模式\n• 建议复核工艺参数" : "• No obvious defect pattern\n• Check process parameters";
    }
}

void MainWindow::calcWaferStats(const cv::Mat& wafer,
                                int& totalDie, int& passDie, int& failDie,
                                float& yieldRate, float& defectDensity,
                                float& cx, float& cy, float& compactness, float& aspectRatio)
{
    totalDie = 0; passDie = 0; failDie = 0;
    std::vector<cv::Point2i> defects;
    for (int y = 0; y < wafer.rows; y++) {
        const uchar* row = wafer.ptr<uchar>(y);
        for (int x = 0; x < wafer.cols; x++) {
            uchar val = row[x];
            if (val > 0) totalDie++;
            if (val == 1) passDie++;
            if (val == 2) { failDie++; defects.push_back(cv::Point2i(x, y)); }
        }
    }
    yieldRate = totalDie > 0 ? (float)passDie / totalDie : 0.0f;
    defectDensity = totalDie > 0 ? (float)failDie / totalDie : 0.0f;
    double sumX = 0, sumY = 0;
    for (auto& p : defects) { sumX += p.x; sumY += p.y; }
    cx = defects.empty() ? 0.5f : (float)(sumX / defects.size() / wafer.cols);
    cy = defects.empty() ? 0.5f : (float)(sumY / defects.size() / wafer.rows);
    if (defects.empty()) {
        compactness = 0.0f; aspectRatio = 1.0f; return;
    }
    int xMin = wafer.cols, xMax = 0, yMin = wafer.rows, yMax = 0;
    for (auto& p : defects) {
        xMin = std::min(xMin, p.x); xMax = std::max(xMax, p.x);
        yMin = std::min(yMin, p.y); yMax = std::max(yMax, p.y);
    }
    int bw = xMax - xMin + 1;
    int bh = yMax - yMin + 1;
    compactness = (bw * bh) > 0 ? (float)defects.size() / (bw * bh) : 0.0f;
    aspectRatio = bw > 0 ? (float)bh / bw : 1.0f;
}

void MainWindow::showWaferByIndex(int globalIndex)
{
    if (globalIndex < 0 || globalIndex >= (int)m_allResults.size()) return;
    const auto& r = m_allResults[globalIndex];
    cv::Mat wafer = loadWaferFile(r.fullPath);
    if (wafer.empty()) return;
    QPixmap waferPix = cvMatToPixmap(wafer);
    QSize shownSize = waferPix.scaled(
                                  ui->labelWaferView->size(), Qt::KeepAspectRatio, Qt::FastTransformation).size();
    ui->labelWaferView->setPixmap(waferPix.scaled(
        shownSize, Qt::KeepAspectRatio, Qt::FastTransformation));
    if (m_showSingleHeat) {
        ui->labelHeatSingle->setPixmap(singleWaferHeatMap(wafer).scaled(
            shownSize, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    bool lowConf = r.isLowConfidence;
    if (r.isNormal) {
        ui->labelResult->setText(m_isZh
                                     ? "晶圆状态：正常\n无明显缺陷模式"
                                     : "Wafer Status: Normal\nNo obvious defect pattern");
    } else {
        ui->labelResult->setText(m_isZh
                                     ? QString("预测类别：%1\n置信度：%2\n次选类别：%3 (%4)%5")
                                           .arg(r.className).arg(r.confidence, 0, 'f', 4)
                                           .arg(r.secondClassName).arg(r.secondConfidence, 0, 'f', 4)
                                           .arg(lowConf ? "\n⚠ 置信度偏低，建议复核" : "")
                                     : QString("Class: %1\nConfidence: %2\n2nd: %3 (%4)%5")
                                           .arg(r.className).arg(r.confidence, 0, 'f', 4)
                                           .arg(r.secondClassName).arg(r.secondConfidence, 0, 'f', 4)
                                           .arg(lowConf ? "\n⚠ Low confidence, review suggested" : ""));
    }
    QString detail;
    if (m_isZh) {
        detail = QString(
                     "━━━ 质量统计 ━━━\n"
                     "总Die数：%1   良品：%2   失效：%3\n"
                     "良率：%4%   失效密度：%5%\n\n"
                     "━━━ 空间分布 ━━━\n"
                     "失效重心：X=%6 , Y=%7\n"
                     "分布紧凑度：%8\n"
                     "区域长宽比：%9\n\n"
                     "━━━ 工艺建议排查 ━━━\n%10"
                     ).arg(r.totalDie).arg(r.passDie).arg(r.failDie)
                     .arg(r.yieldRate*100, 0, 'f', 2).arg(r.defectDensity*100, 0, 'f', 2)
                     .arg(r.cx, 0, 'f', 3).arg(r.cy, 0, 'f', 3)
                     .arg(r.compactness, 0, 'f', 3)
                     .arg(r.aspectRatio, 0, 'f', 3)
                     .arg(getProcessSuggestion(r.classId));
    } else {
        detail = QString(
                     "━━━ Quality Stats ━━━\n"
                     "Total: %1  Pass: %2  Fail: %3\n"
                     "Yield: %4%   Defect Density: %5%\n\n"
                     "━━━ Spatial Dist. ━━━\n"
                     "Center: X=%6 , Y=%7\n"
                     "Compactness: %8\n"
                     "Aspect Ratio: %9\n\n"
                     "━━━ Process Check ━━━\n%10"
                     ).arg(r.totalDie).arg(r.passDie).arg(r.failDie)
                     .arg(r.yieldRate*100, 0, 'f', 2).arg(r.defectDensity*100, 0, 'f', 2)
                     .arg(r.cx, 0, 'f', 3).arg(r.cy, 0, 'f', 3)
                     .arg(r.compactness, 0, 'f', 3)
                     .arg(r.aspectRatio, 0, 'f', 3)
                     .arg(getProcessSuggestion(r.classId));
    }
    ui->labelDetail->setText(detail);
    updateNavLabel();
}

void MainWindow::updateNavLabel()
{
    int total = (int)m_allResults.size();
    int current = m_currentIndex + 1;
    if (total == 0) current = 0;
    ui->labelNav->setText(m_isZh
                              ? QString("第 %1 / %2 片").arg(current).arg(total)
                              : QString("%1 / %2 Wafers").arg(current).arg(total));
}

// ==================== 文件读取 ====================
cv::Mat MainWindow::loadWaferFile(const QString& filePath)
{
    QString suffix = QFileInfo(filePath).suffix().toLower();
    cv::Mat out;
    if (suffix == "bin") {
        out.create(64, 64, CV_8UC1);
        FILE* fp = fopen(filePath.toStdString().c_str(), "rb");
        if (!fp) return cv::Mat();
        fread(out.data, 1, 64 * 64, fp);
        fclose(fp);
        return out;
    }
    else if (suffix == "png" || suffix == "jpg" || suffix == "bmp") {
        cv::Mat src = cv::imread(filePath.toStdString(), cv::IMREAD_GRAYSCALE);
        if (src.empty()) return cv::Mat();
        cv::resize(src, out, cv::Size(64, 64), 0, 0, cv::INTER_NEAREST);
        cv::Mat triVal(64, 64, CV_8UC1);
        for (int y = 0; y < 64; y++) {
            const uchar* s = out.ptr<uchar>(y);
            uchar* d = triVal.ptr<uchar>(y);
            for (int x = 0; x < 64; x++) {
                uchar v = s[x];
                if (v < 45) d[x] = 0;
                else if (v < 100) d[x] = 2;
                else d[x] = 1;
            }
        }
        return triVal;
    }
    else if (suffix == "stdf") {
        try {
            StdfParser parser;
            cv::Mat rawWafer = parser.parse(filePath.toStdString());
            if (rawWafer.empty()) {
                appendLog(m_isZh ? "[错误] STDF解析失败，格式不兼容" : "[ERROR] STDF parse failed");
                return cv::Mat();
            }
            cv::resize(rawWafer, out, cv::Size(64, 64), 0, 0, cv::INTER_NEAREST);
            return out;
        } catch (...) {
            appendLog(m_isZh ? "[错误] STDF文件损坏" : "[ERROR] STDF file corrupted");
            return cv::Mat();
        }
    }
    return cv::Mat();
}

QPixmap MainWindow::cvMatToPixmap(const cv::Mat& mat)
{
    cv::Mat color(mat.rows, mat.cols, CV_8UC3);
    for (int y = 0; y < mat.rows; y++) {
        const uchar* row = mat.ptr<uchar>(y);
        cv::Vec3b* dstRow = color.ptr<cv::Vec3b>(y);
        for (int x = 0; x < mat.cols; x++) {
            uchar val = row[x];
            if (val == 2)      dstRow[x] = cv::Vec3b(0, 0, 255);
            else if (val == 1) dstRow[x] = cv::Vec3b(0, 200, 0);
            else               dstRow[x] = cv::Vec3b(30, 30, 30);
        }
    }
    QImage img(color.data, color.cols, color.rows, (int)color.step, QImage::Format_RGB888);
    return QPixmap::fromImage(img.rgbSwapped());
}

void MainWindow::appendLog(const QString& msg)
{
    ui->textLog->append(msg);
}

void MainWindow::refreshUiText()
{
    ui->comboLang->blockSignals(true);
    if (m_isZh) {
        ui->btnSingleOpen->setText("打开单文件推理");
        ui->btnBatch->setText("批量文件夹推理");
        ui->btnExport->setText("导出CSV结果");
        ui->btnGlobalHeat->setText("全局热力图");
        ui->btnRole->setText("角色");
        ui->cbShowHeatSingle->setText("显示单图热力");
        ui->cbTTA->setText("启用TTA测试时增强");
        ui->labelResultTitle->setText("识别结果");
        ui->labelLogTitle->setText("运行日志");
        ui->btnPrev->setText("上一张 (A)");
        ui->btnNext->setText("下一张 (D)");
        ui->cbFilterLow->setText("仅显示异常");
        ui->comboFilterType->blockSignals(true);
        ui->comboFilterType->clear();
        ui->comboFilterType->addItem("全部类型");
        ui->comboFilterType->addItem("仅单次");
        ui->comboFilterType->addItem("仅批量");
        ui->comboFilterType->setCurrentIndex(0);
        ui->comboFilterType->blockSignals(false);
        ui->comboFilterClass->blockSignals(true);
        ui->comboFilterClass->clear();
        ui->comboFilterClass->addItem("全部类别");
        ui->comboFilterClass->addItem("Center");
        ui->comboFilterClass->addItem("Donut");
        ui->comboFilterClass->addItem("Edge-Loc");
        ui->comboFilterClass->addItem("Edge-Ring");
        ui->comboFilterClass->addItem("Loc");
        ui->comboFilterClass->addItem("Random");
        ui->comboFilterClass->addItem("Scratch");
        ui->comboFilterClass->addItem("Near-full");
        ui->comboFilterClass->addItem("正常");
        ui->comboFilterClass->setCurrentIndex(0);
        ui->comboFilterClass->blockSignals(false);
        ui->comboLang->clear();
        ui->comboLang->addItem("中文");
        ui->comboLang->addItem("English");
        ui->comboLang->setCurrentIndex(0);
    } else {
        ui->btnSingleOpen->setText("Open Single File");
        ui->btnBatch->setText("Batch Inference");
        ui->btnExport->setText("Export CSV");
        ui->btnGlobalHeat->setText("Global Heatmap");
        ui->btnRole->setText("Role");
        ui->cbShowHeatSingle->setText("Show Single Heat");
        ui->cbTTA->setText("Enable TTA Augmentation");
        ui->labelResultTitle->setText("Results");
        ui->labelLogTitle->setText("Log");
        ui->btnPrev->setText("Prev (A)");
        ui->btnNext->setText("Next (D)");
        ui->cbFilterLow->setText("Low confidence only");
        ui->comboFilterType->blockSignals(true);
        ui->comboFilterType->clear();
        ui->comboFilterType->addItem("All Types");
        ui->comboFilterType->addItem("Single only");
        ui->comboFilterType->addItem("Batch only");
        ui->comboFilterType->setCurrentIndex(0);
        ui->comboFilterType->blockSignals(false);
        ui->comboFilterClass->blockSignals(true);
        ui->comboFilterClass->clear();
        ui->comboFilterClass->addItem("All Classes");
        ui->comboFilterClass->addItem("Center");
        ui->comboFilterClass->addItem("Donut");
        ui->comboFilterClass->addItem("Edge-Loc");
        ui->comboFilterClass->addItem("Edge-Ring");
        ui->comboFilterClass->addItem("Loc");
        ui->comboFilterClass->addItem("Random");
        ui->comboFilterClass->addItem("Scratch");
        ui->comboFilterClass->addItem("Near-full");
        ui->comboFilterClass->addItem("Normal");
        ui->comboFilterClass->setCurrentIndex(0);
        ui->comboFilterClass->blockSignals(false);
        ui->comboLang->clear();
        ui->comboLang->addItem("中文");
        ui->comboLang->addItem("English");
        ui->comboLang->setCurrentIndex(1);
    }
    ui->comboLang->blockSignals(false);

    // ===== 菜单栏 / 后端下拉 / 设备面板 双语 =====
    m_menuFile->setTitle(m_isZh ? "文件(&F)" : "File(&F)");
    m_menuView->setTitle(m_isZh ? "视图(&V)" : "View(&V)");
    m_menuHelp->setTitle(m_isZh ? "帮助(&H)" : "Help(&H)");
    m_actOpenSingle->setText(m_isZh ? "打开单文件推理" : "Open Single File");
    m_actBatch->setText(m_isZh ? "批量文件夹推理" : "Batch Inference");
    m_actExportCsv->setText(m_isZh ? "导出CSV结果" : "Export CSV");
    m_actExit->setText(m_isZh ? "退出" : "Exit");
    m_actGlobalHeat->setText(m_isZh ? "全局热力图" : "Global Heatmap");
    m_actSingleHeat->setText(m_isZh ? "显示单图热力" : "Show Single Heat");
    m_actSingleHeat->setChecked(m_showSingleHeat);
    m_actAbout->setText(m_isZh ? "关于" : "About");
    if (m_actSwitchMode) m_actSwitchMode->setText(m_isZh ? "切换模式..." : "Switch Mode...");
    m_backendLabel->setText(m_isZh ? "推理后端：" : "Backend:");
    m_comboBackend->blockSignals(true);
    m_comboBackend->setItemText(0, m_isZh ? "本地CPU" : "Local CPU");
    m_comboBackend->setItemText(1, m_isZh ? "RV1106边缘NPU" : "RV1106 Edge NPU");
    m_comboBackend->blockSignals(false);
    m_deviceBtn->setText(m_isZh ? "设备连接" : "Device");
    m_webBtn->setText(m_isZh ? "网页看板" : "Web Dashboard");
    m_exportBtn->setText(m_isZh ? "导出并下发" : "Export & Upload");
    if (m_discoveredIp.isEmpty()) {
        m_discoveryLabel->setText(m_isZh ? "未发现设备" : "No device");
        m_discoveryLabel->setStyleSheet("color:#888;font-size:11px;padding:0 8px;");
    } else {
        m_discoveryLabel->setText(m_isZh ? QString("已发现: %1").arg(m_discoveredIp)
                                         : QString("Found: %1").arg(m_discoveredIp));
        m_discoveryLabel->setStyleSheet("color:#4caf50;font-size:11px;padding:0 8px;");
    }
    m_devicePanel->setLanguage(m_isZh);
    // 同步结果面板语言
    if (m_currentIndex < 0) {
        ui->labelResult->setText(m_isZh ? "等待推理..." : "Waiting...");
    } else {
        showWaferByIndex(m_currentIndex);
    }

    buildDisplayRows();
    fillResultTable();
    updateNavLabel();
    applyRolePermission();
}

// ==================== 新增：菜单栏构建 ====================
void MainWindow::buildMenuBar()
{
    m_menuFile = menuBar()->addMenu("文件(&F)");
    m_actOpenSingle = m_menuFile->addAction("打开单文件推理");
    m_actBatch      = m_menuFile->addAction("批量文件夹推理");
    m_menuFile->addSeparator();
    m_actExportCsv   = m_menuFile->addAction("导出CSV结果");
    m_menuFile->addSeparator();
    m_actSwitchMode = m_menuFile->addAction(m_isZh ? "切换模式..." : "Switch Mode...");
    connect(m_actSwitchMode, &QAction::triggered, this, &MainWindow::slot_switchMode);
    m_menuFile->addSeparator();
    m_actExit        = m_menuFile->addAction("退出");

    m_menuView = menuBar()->addMenu("视图(&V)");
    m_actGlobalHeat  = m_menuView->addAction("全局热力图");
    m_actSingleHeat  = m_menuView->addAction("显示单图热力");
    m_actSingleHeat->setCheckable(true);

    m_menuHelp = menuBar()->addMenu("帮助(&H)");
    m_actAbout      = m_menuHelp->addAction("关于");

    connect(m_actOpenSingle, &QAction::triggered, this, &MainWindow::slot_openSingleFile);
    connect(m_actBatch,      &QAction::triggered, this, &MainWindow::slot_batchInfer);
    connect(m_actExportCsv,   &QAction::triggered, this, &MainWindow::slot_exportCsv);
    connect(m_actExit,        &QAction::triggered, this, &QWidget::close);
    connect(m_actGlobalHeat,  &QAction::triggered, this, &MainWindow::slot_toggleGlobalHeat);
    connect(m_actSingleHeat,  &QAction::triggered, this, &MainWindow::slot_toggleSingleHeat);
    connect(m_actAbout,       &QAction::triggered, this, &MainWindow::slot_about);
}

// ==================== 新增：后端切换槽函数 ====================
void MainWindow::slot_switchBackend(int idx)
{
    m_useEdgeNpu = (idx == 1);
    if (m_useEdgeNpu && !m_devicePanel->isConnected()) {
        int ret = QMessageBox::question(this,
                                        m_isZh ? "边缘推理" : "Edge Inference",
                                        m_isZh ? "启用边缘 NPU 推理需要连接设备，是否现在连接？"
                                               : "Edge NPU inference requires a device. Connect now?",
                                        QMessageBox::Yes | QMessageBox::No);
        if (ret == QMessageBox::Yes) {
            slot_openDeviceDialog();
        } else {
            m_comboBackend->blockSignals(true);
            m_comboBackend->setCurrentIndex(0);
            m_comboBackend->blockSignals(false);
            m_useEdgeNpu = false;
        }
    }
    appendLog(m_isZh
                  ? QString("推理后端切换为：%1").arg(m_useEdgeNpu ? "RV1106边缘NPU" : "本地CPU")
                  : QString("Backend switched to: %1").arg(m_useEdgeNpu ? "RV1106 Edge NPU" : "Local CPU"));
}

void MainWindow::slot_openDeviceDialog()
{
    m_devicePanel->exec();
}

void MainWindow::slot_about()
{
    if (m_isZh) {
        QMessageBox::about(this, "关于",
                           "基于边缘AI的晶圆缺陷检测上位机软件 V1.0\n\n"
                           "支持本地CPU推理与RV1106边缘NPU加速推理\n"
                           "支持晶圆缺陷模式识别、热力图分析、批量检测\n\n"
                           "电子科技大学 嵌入式系统专业");
    } else {
        QMessageBox::about(this, "About",
                           "Edge-AI Wafer Defect Inspection Host Software V1.0\n\n"
                           "Local CPU inference and RV1106 edge NPU accelerated inference\n"
                           "Wafer defect pattern recognition, heatmap analysis, batch inspection\n\n"
                           "UESTC, Embedded Systems");
    }
}

// ==================== 新增：边缘NPU推理核心函数 ====================
bool MainWindow::edgeInfer(const cv::Mat& wafer,
                           int& classId, float& confidence,
                           std::array<float, 8>& probs)
{
    if (!m_devicePanel->isConnected()) {
        appendLog(m_isZh ? "[错误] 边缘设备未连接" : "[ERR] Edge device not connected");
        return false;
    }
    QTcpSocket *socket = m_devicePanel->socket();

    // ===== 1. 构造发送包：magic(4) + 图像NHWC + 特征float32 =====
    const int imgBytes = 64 * 64 * 3;
    const int featBytes = 16 * sizeof(float);
    const int sendSize = 4 + imgBytes + featBytes;
    QByteArray sendBuf(sendSize, 0);
    char *ptr = sendBuf.data();

    memcpy(ptr, "RFNS", 4);
    ptr += 4;

    // 图像：NHWC uint8 三通道（失效/良品/背景分别对应RGB通道）
    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            uchar val = wafer.at<uchar>(y, x);
            uchar r = (val == 2) ? 255 : 0;
            uchar g = (val == 1) ? 255 : 0;
            uchar b = (val == 0) ? 255 : 0;
            int pixIdx = (y * 64 + x) * 3;
            ptr[pixIdx + 0] = r;
            ptr[pixIdx + 1] = g;
            ptr[pixIdx + 2] = b;
        }
    }
    ptr += imgBytes;

    // 16维特征：float32 小端
    WaferFeature featExt;
    featExt.setNorm(FEAT_MEAN, FEAT_STD);
    WaferFeature::FeatureVec feat = featExt.extract(wafer);
    for (int i = 0; i < 16; i++) {
        float f = feat[i];
        memcpy(ptr, &f, sizeof(float));
        ptr += sizeof(float);
    }

    // 发送
    socket->write(sendBuf);
    if (!socket->waitForBytesWritten(3000)) {
        appendLog(m_isZh ? "[错误] 发送数据超时" : "[ERR] Send data timeout");
        return false;
    }

    // ===== 2. 接收回包：magic(4) + pred(int32) + conf(float32) + 8*logits =====
    const int recvSize = 4 + 4 + 4 + 8 * (int)sizeof(float);
    if (!socket->waitForReadyRead(5000)) {
        appendLog(m_isZh ? "[错误] 接收结果超时" : "[ERR] Recv result timeout");
        return false;
    }
    QByteArray recvBuf = socket->readAll();
    while (recvBuf.size() < recvSize) {
        if (!socket->waitForReadyRead(1000)) {
            appendLog(m_isZh ? "[错误] 接收数据不完整" : "[ERR] Incomplete recv data");
            return false;
        }
        recvBuf += socket->readAll();
    }

    const char *rptr = recvBuf.data();
    char magic[5] = {0};
    memcpy(magic, rptr, 4);
    if (strcmp(magic, "RFNR") != 0) {
        appendLog(m_isZh ? "[错误] 回包校验失败" : "[ERR] Reply magic mismatch");
        return false;
    }
    rptr += 4;

    int32_t pred;
    memcpy(&pred, rptr, sizeof(int32_t));
    rptr += sizeof(int32_t);

    float conf;
    memcpy(&conf, rptr, sizeof(float));
    rptr += sizeof(float);

    std::array<float, 8> logits;
    for (int i = 0; i < 8; i++) {
        memcpy(&logits[i], rptr, sizeof(float));
        rptr += sizeof(float);
    }

    // softmax 转概率
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    float sumExp = 0.0f;
    for (int i = 0; i < 8; i++) {
        probs[i] = std::exp(logits[i] - maxLogit);
        sumExp += probs[i];
    }
    for (int i = 0; i < 8; i++) probs[i] /= sumExp;

    classId = pred;
    confidence = conf;
    return true;
}

// ==================== 打开网页看板 ====================
void MainWindow::slot_openWebDashboard()
{
    // 优先用UDP自动发现的IP (最准确), 其次设备面板, 最后兜底
    QString ip = m_discoveredIp;
    if (ip.isEmpty()) ip = m_devicePanel->getIp();
    if (ip.isEmpty()) {
        bool okIp;
        ip = QInputDialog::getText(this,
                                   m_isZh ? "输入板子 IP" : "Enter Board IP",
                                   m_isZh ? "未自动发现设备，请输入板子 IP:" : "No device found, enter board IP:",
                                   QLineEdit::Normal, "192.168.1.100", &okIp);
        if (!okIp || ip.isEmpty()) return;
    }
    QString url = QString("http://%1:8081").arg(ip);
    QDesktopServices::openUrl(QUrl(url));
    appendLog(m_isZh ? QString("[信息] 已打开网页看板: %1").arg(url)
                     : QString("[INFO] Web dashboard opened: %1").arg(url));
}

// ==================== 导出并下发到板端直解 ====================
void MainWindow::slot_exportAndUpload()
{
    // 1. 选择文件 (支持多选, 混合格式)
    QStringList files = QFileDialog::getOpenFileNames(this,
                                                      m_isZh ? "选择晶圆数据文件" : "Select Wafer Data Files",
                                                      QString(),
                                                      "Wafer Data (*.bin *.png *.stdf);;All Files (*.*)");
    if (files.isEmpty()) return;
    if (m_prodLog) m_prodLog->append(m_isZh ? QString("[开始] 选择 %1 个文件").arg(files.size())
                                 : QString("[START] %1 files selected").arg(files.size()));

    // 2. 获取板子 IP: 优先 UDP 自动发现 → 设备面板 → 手动输入
    QString ip = m_discoveredIp;
    if (ip.isEmpty()) ip = m_devicePanel->getIp();
    if (ip.isEmpty()) {
        bool okIp;
        ip = QInputDialog::getText(this,
                                   m_isZh ? "输入板子 IP" : "Enter Board IP",
                                   m_isZh ? "未自动发现设备，请输入板子 IP:" : "No device found, enter board IP:",
                                   QLineEdit::Normal, "192.168.1.100", &okIp);
        if (!okIp || ip.isEmpty()) return;
    }

    // 3. 本地保存目录
    QDir().mkpath("edge_bin");

    int ok = 0, fail = 0;
    QNetworkAccessManager *nam = new QNetworkAccessManager(this);

    for (const QString& fpath : files) {
        QFileInfo fi(fpath);
        QString baseName = fi.completeBaseName();

        // 4. C++ 内部转换 (无 Python 依赖)
        QByteArray edgeBin = WaferConverter::convertAuto(fpath);
        if (edgeBin.size() != WaferConverter::EDGE_BIN_SIZE) {
            appendLog(m_isZh ? QString("[失败] 转换失败: %1").arg(fi.fileName())
                             : QString("[FAIL] Convert failed: %1").arg(fi.fileName()));
            if (m_prodLog) m_prodLog->append(m_isZh ? QString("[失败] 转换失败: %1").arg(fi.fileName())
                                         : QString("[FAIL] Convert: %1").arg(fi.fileName()));
            fail++;
            continue;
        }

        // 5. 本地保存一份
        QString localPath = QString("edge_bin/%1_edge.bin").arg(baseName);
        WaferConverter::saveEdgeBin(edgeBin, localPath);

        // 6. HTTP POST 到板子 incoming/
        QString upName = QString("%1_edge.bin").arg(baseName);
        QUrl url(QString("http://%1:8081/api/upload?name=%2").arg(ip, upName));
        QNetworkRequest req(url);
        req.setHeader(QNetworkRequest::ContentTypeHeader, "application/octet-stream");
        req.setHeader(QNetworkRequest::ContentLengthHeader, edgeBin.size());

        QNetworkReply *reply = nam->post(req, edgeBin);
        QEventLoop loop;
        connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
        loop.exec();

        if (reply->error() == QNetworkReply::NoError) {
            appendLog(m_isZh ? QString("[下发] %1 → 板子 incoming/ (12352B)").arg(upName)
                             : QString("[UPLOAD] %1 → board incoming/ (12352B)").arg(upName));
            if (m_prodLog) m_prodLog->append(m_isZh ? QString("[下发] %1 OK").arg(upName)
                                         : QString("[UPLOAD] %1 OK").arg(upName));
            ok++;
        } else {
            appendLog(m_isZh ? QString("[失败] 下发失败: %1 (%2)").arg(upName, reply->errorString())
                             : QString("[FAIL] Upload failed: %1 (%2)").arg(upName, reply->errorString()));
            if (m_prodLog) m_prodLog->append(m_isZh ? QString("[失败] %1: %2").arg(upName, reply->errorString())
                                         : QString("[FAIL] %1: %2").arg(upName, reply->errorString()));
            fail++;
        }
        reply->deleteLater();
    }

    appendLog(m_isZh ? QString("[汇总] 导出并下发完成: 成功 %1, 失败 %2").arg(ok).arg(fail)
                     : QString("[SUMMARY] Export & Upload done: %1 ok, %2 failed").arg(ok).arg(fail));
    if (m_prodLog) m_prodLog->append(m_isZh ? QString("[汇总] 成功 %1, 失败 %2").arg(ok).arg(fail)
                                 : QString("[DONE] %1 ok, %2 failed").arg(ok).arg(fail));
}

// ==================== UDP 自动发现板子 ====================
void MainWindow::slot_udpDiscovery()
{
    while (m_udpSocket->hasPendingDatagrams()) {
        QByteArray datagram;
        datagram.resize(m_udpSocket->pendingDatagramSize());
        QHostAddress sender;
        quint16 senderPort;
        m_udpSocket->readDatagram(datagram.data(), datagram.size(), &sender, &senderPort);

        QString msg = QString::fromUtf8(datagram);
        // 解析 JSON: {"type":"wafer_direct","ip":"10.125.98.193","http":8081,...}
        if (msg.contains("\"type\":\"wafer_direct\"")) {
            int ipStart = msg.indexOf("\"ip\":\"") + 6;
            int ipEnd = msg.indexOf("\"", ipStart);
            if (ipStart > 6 && ipEnd > ipStart) {
                QString ip = msg.mid(ipStart, ipEnd - ipStart);
                if (ip != m_discoveredIp) {
                    m_discoveredIp = ip;
                    if (m_discoveryTimer) m_discoveryTimer->stop();
                    m_discoveryLabel->setText(m_isZh ? QString("已发现: %1").arg(ip)
                                                     : QString("Found: %1").arg(ip));
                    m_discoveryLabel->setStyleSheet("color:#4caf50;font-size:11px;padding:0 8px;");
                    // 自动填入设备连接面板的 IP 框
                    m_devicePanel->setIp(ip);
                    appendLog(m_isZh ? QString("[发现] 板子自动发现: %1").arg(ip)
                                     : QString("[DISCOVERY] Board found: %1").arg(ip));
                    if (m_prodLog) m_prodLog->append(m_isZh ? QString("[发现] 板子: %1").arg(ip)
                                                 : QString("[FOUND] Board: %1").arg(ip));
                    // 产线模式下同步更新面板状态
                    if (m_currentMode == MODE_PRODUCTION && m_prodStatusLabel) {
                        m_prodStatusLabel->setText(m_isZh ? QString("已连接板子: %1").arg(ip)
                                                          : QString("Board connected: %1").arg(ip));
                        m_prodStatusLabel->setStyleSheet("font-size:14px;color:#4caf50;padding:8px;");
                        if (m_prodRetryBtn) m_prodRetryBtn->hide();
                    }
                }
            }
        }
    }
}

// ==================== 启动模式选择对话框 ====================
AppMode MainWindow::showModeSelectDialog()
{
    QDialog dlg(this);
    dlg.setWindowTitle(m_isZh ? "选择运行模式" : "Select Mode");
    dlg.setFixedSize(560, 360);
    dlg.setStyleSheet("QDialog{background:#f5f5f5;}");

    QVBoxLayout *mainLayout = new QVBoxLayout(&dlg);
    mainLayout->setContentsMargins(30, 25, 30, 20);

    // 顶栏: 标题 + 语言切换
    QHBoxLayout *topRow = new QHBoxLayout();
    QLabel *title = new QLabel(m_isZh ? "请选择运行模式" : "Please Select Mode", &dlg);
    title->setStyleSheet("font-size:18px;font-weight:bold;color:#333;");
    title->setAlignment(Qt::AlignCenter);
    topRow->addWidget(title, 1);
    QPushButton *langBtn = new QPushButton(m_isZh ? "EN" : "中", &dlg);
    langBtn->setFixedSize(48, 30);
    langBtn->setStyleSheet(
        "QPushButton{background:#fff;color:#333;border:1px solid #ccc;border-radius:4px;font-size:13px;}"
        "QPushButton:hover{background:#eee;}");
    topRow->addWidget(langBtn, 0, Qt::AlignRight);
    mainLayout->addLayout(topRow);
    mainLayout->addSpacing(15);

    QHBoxLayout *cardLayout = new QHBoxLayout();
    cardLayout->setSpacing(20);

    // ── 研发模式卡片 ──
    QPushButton *btnRnd = new QPushButton(&dlg);
    btnRnd->setFixedSize(230, 200);
    btnRnd->setStyleSheet(
        "QPushButton{background:#fff;border:2px solid #ddd;border-radius:10px;}"
        "QPushButton:hover{border:2px solid #2196f3;background:#e3f2fd;}");
    QVBoxLayout *rndLayout = new QVBoxLayout(btnRnd);
    rndLayout->setAlignment(Qt::AlignCenter);
    QLabel *rndIcon = new QLabel("\xe2\x9c\x8f", btnRnd);  // wrench emoji
    rndIcon->setStyleSheet("font-size:42px;");
    rndIcon->setAlignment(Qt::AlignCenter);
    QLabel *rndTitle = new QLabel(m_isZh ? "研发调试模式" : "R&D Mode", btnRnd);
    rndTitle->setStyleSheet("font-size:16px;font-weight:bold;color:#2196f3;");
    rndTitle->setAlignment(Qt::AlignCenter);
    QLabel *rndDesc = new QLabel(
        m_isZh ? "本地CPU / 边缘NPU推理\n交互式分析\n结果显示在软件内"
               : "Local CPU / Edge NPU\nInteractive analysis\nResults in app", btnRnd);
    rndDesc->setStyleSheet("font-size:11px;color:#666;");
    rndDesc->setAlignment(Qt::AlignCenter);
    rndLayout->addWidget(rndIcon);
    rndLayout->addSpacing(8);
    rndLayout->addWidget(rndTitle);
    rndLayout->addSpacing(5);
    rndLayout->addWidget(rndDesc);

    // ── 产线直解模式卡片 ──
    QPushButton *btnProd = new QPushButton(&dlg);
    btnProd->setFixedSize(230, 200);
    btnProd->setStyleSheet(
        "QPushButton{background:#fff;border:2px solid #ddd;border-radius:10px;}"
        "QPushButton:hover{border:2px solid #4caf50;background:#e8f5e9;}");
    QVBoxLayout *prodLayout = new QVBoxLayout(btnProd);
    prodLayout->setAlignment(Qt::AlignCenter);
    QLabel *prodIcon = new QLabel("\xf0\x9f\x8f\xad", btnProd);  // factory emoji
    prodIcon->setStyleSheet("font-size:42px;");
    prodIcon->setAlignment(Qt::AlignCenter);
    QLabel *prodTitle = new QLabel(m_isZh ? "产线直解模式" : "Production Mode", btnProd);
    prodTitle->setStyleSheet("font-size:16px;font-weight:bold;color:#4caf50;");
    prodTitle->setAlignment(Qt::AlignCenter);
    QLabel *prodDesc = new QLabel(
        m_isZh ? "文件上传到板子\n板子自动推理\n网页实时监控"
               : "Upload files to board\nAuto inference\nWeb dashboard", btnProd);
    prodDesc->setStyleSheet("font-size:11px;color:#666;");
    prodDesc->setAlignment(Qt::AlignCenter);
    prodLayout->addWidget(prodIcon);
    prodLayout->addSpacing(8);
    prodLayout->addWidget(prodTitle);
    prodLayout->addSpacing(5);
    prodLayout->addWidget(prodDesc);

    cardLayout->addWidget(btnRnd);
    cardLayout->addWidget(btnProd);
    mainLayout->addLayout(cardLayout);

    AppMode result = MODE_RND;
    connect(btnRnd, &QPushButton::clicked, [&](){ result = MODE_RND; dlg.accept(); });
    connect(btnProd, &QPushButton::clicked, [&](){ result = MODE_PRODUCTION; dlg.accept(); });

    // 弹窗内语言切换: 即时更新所有文字
    connect(langBtn, &QPushButton::clicked, [&](){
        m_isZh = !m_isZh;
        dlg.setWindowTitle(m_isZh ? "选择运行模式" : "Select Mode");
        langBtn->setText(m_isZh ? "EN" : "中");
        title->setText(m_isZh ? "请选择运行模式" : "Please Select Mode");
        rndTitle->setText(m_isZh ? "研发调试模式" : "R&D Mode");
        rndDesc->setText(m_isZh ? "本地CPU / 边缘NPU推理\n交互式分析\n结果显示在软件内"
                                : "Local CPU / Edge NPU\nInteractive analysis\nResults in app");
        prodTitle->setText(m_isZh ? "产线直解模式" : "Production Mode");
        prodDesc->setText(m_isZh ? "文件上传到板子\n板子自动推理\n网页实时监控"
                                 : "Upload files to board\nAuto inference\nWeb dashboard");
    });

    dlg.exec();
    return result;
}

// ==================== 构建产线模式面板 ====================
void MainWindow::buildProductionPanel()
{
    if (m_prodWidget) return;
    m_prodWidget = new QWidget(centralWidget());
    m_prodWidget->setStyleSheet("background:#fafafa;");

    QVBoxLayout *layout = new QVBoxLayout(m_prodWidget);
    layout->setAlignment(Qt::AlignCenter);
    layout->setSpacing(15);
    layout->setContentsMargins(40, 30, 40, 30);

    // 顶栏: 标题 + 语言切换
    QHBoxLayout *topBar = new QHBoxLayout();
    m_prodTitleLabel = new QLabel(m_isZh ? "产线直解模式" : "Production Direct Mode", m_prodWidget);
    m_prodTitleLabel->setStyleSheet("font-size:24px;font-weight:bold;color:#2e7d32;");
    m_prodTitleLabel->setAlignment(Qt::AlignCenter);
    topBar->addWidget(m_prodTitleLabel, 1);
    m_prodLangBtn = new QPushButton(m_isZh ? "EN" : "中", m_prodWidget);
    m_prodLangBtn->setFixedSize(48, 32);
    m_prodLangBtn->setStyleSheet(
        "QPushButton{background:#fff;color:#2e7d32;border:1px solid #2e7d32;border-radius:4px;font-size:13px;}"
        "QPushButton:hover{background:#e8f5e9;}");
    connect(m_prodLangBtn, &QPushButton::clicked, this, &MainWindow::slot_prodToggleLang);
    topBar->addWidget(m_prodLangBtn, 0, Qt::AlignRight);
    layout->addLayout(topBar);

    m_prodDescLabel = new QLabel(
        m_isZh ? "选择晶圆数据文件（.bin / .png / .stdf），自动转换并上传到板子\n板子自动推理，结果实时推送到网页看板"
               : "Select wafer data files (.bin / .png / .stdf)\nAuto convert & upload to board, results on web dashboard",
        m_prodWidget);
    m_prodDescLabel->setStyleSheet("font-size:13px;color:#555;");
    m_prodDescLabel->setAlignment(Qt::AlignCenter);
    layout->addWidget(m_prodDescLabel);

    // 设备状态 + 重试按钮
    QHBoxLayout *statusRow = new QHBoxLayout();
    statusRow->setAlignment(Qt::AlignCenter);
    m_prodStatusLabel = new QLabel(m_isZh ? "等待发现板子..." : "Waiting for board...", m_prodWidget);
    m_prodStatusLabel->setStyleSheet("font-size:14px;color:#888;padding:8px;");
    m_prodStatusLabel->setAlignment(Qt::AlignCenter);
    statusRow->addWidget(m_prodStatusLabel);
    m_prodRetryBtn = new QPushButton(m_isZh ? "重新发现" : "Retry", m_prodWidget);
    m_prodRetryBtn->setFixedSize(90, 30);
    m_prodRetryBtn->setStyleSheet(
        "QPushButton{background:#fff;color:#e74c3c;border:1px solid #e74c3c;border-radius:4px;font-size:12px;}"
        "QPushButton:hover{background:#fdecea;}");
    m_prodRetryBtn->hide();
    connect(m_prodRetryBtn, &QPushButton::clicked, this, [this](){
        m_discoveredIp.clear();
        if (m_prodStatusLabel) {
            m_prodStatusLabel->setText(m_isZh ? "正在重新发现板子..." : "Rediscovering board...");
            m_prodStatusLabel->setStyleSheet("font-size:14px;color:#f39c12;padding:8px;");
        }
        if (m_discoveryTimer) m_discoveryTimer->start(5000);
    });
    statusRow->addWidget(m_prodRetryBtn);
    layout->addLayout(statusRow);

    // 上传按钮
    m_prodUploadBtn = new QPushButton(m_isZh ? "选择文件并下发" : "Select & Upload Files", m_prodWidget);
    m_prodUploadBtn->setFixedSize(280, 56);
    m_prodUploadBtn->setStyleSheet(
        "QPushButton{background:#4caf50;color:#fff;font-size:16px;font-weight:bold;"
        "border-radius:8px;border:none;}"
        "QPushButton:hover{background:#43a047;}"
        "QPushButton:pressed{background:#388e3c;}");
    connect(m_prodUploadBtn, &QPushButton::clicked, this, &MainWindow::slot_exportAndUpload);
    layout->addWidget(m_prodUploadBtn, 0, Qt::AlignCenter);

    // 网页看板按钮
    m_prodWebBtn = new QPushButton(m_isZh ? "打开网页看板" : "Open Web Dashboard", m_prodWidget);
    m_prodWebBtn->setFixedSize(280, 44);
    m_prodWebBtn->setStyleSheet(
        "QPushButton{background:#fff;color:#2196f3;font-size:14px;"
        "border:2px solid #2196f3;border-radius:8px;}"
        "QPushButton:hover{background:#e3f2fd;}");
    connect(m_prodWebBtn, &QPushButton::clicked, this, &MainWindow::slot_openWebDashboard);
    layout->addWidget(m_prodWebBtn, 0, Qt::AlignCenter);

    // 文件夹监控按钮 + 状态
    m_prodWatchBtn = new QPushButton(m_isZh ? "设置监控文件夹" : "Set Watch Folder", m_prodWidget);
    m_prodWatchBtn->setFixedSize(280, 40);
    m_prodWatchBtn->setStyleSheet(
        "QPushButton{background:#fff;color:#ff9800;font-size:14px;"
        "border:2px solid #ff9800;border-radius:8px;}"
        "QPushButton:hover{background:#fff3e0;}");
    connect(m_prodWatchBtn, &QPushButton::clicked, this, &MainWindow::slot_setWatchFolder);
    layout->addWidget(m_prodWatchBtn, 0, Qt::AlignCenter);

    m_prodWatchLabel = new QLabel(m_isZh ? "未设置监控文件夹" : "No watch folder set", m_prodWidget);
    m_prodWatchLabel->setStyleSheet("font-size:12px;color:#999;");
    m_prodWatchLabel->setAlignment(Qt::AlignCenter);
    m_prodWatchLabel->setWordWrap(true);
    layout->addWidget(m_prodWatchLabel);

    // 日志框
    m_prodLogLabel = new QLabel(m_isZh ? "运行日志" : "Log", m_prodWidget);
    m_prodLogLabel->setStyleSheet("font-size:12px;color:#888;font-weight:bold;");
    layout->addWidget(m_prodLogLabel);
    m_prodLog = new QTextEdit(m_prodWidget);
    m_prodLog->setReadOnly(true);
    m_prodLog->setMaximumHeight(160);
    m_prodLog->setStyleSheet("background:#fff;border:1px solid #ddd;border-radius:4px;font-family:Consolas,monospace;font-size:11px;");
    layout->addWidget(m_prodLog);

    layout->addStretch();
    m_prodWidget->hide();

    // 发现超时定时器: 5秒没发现板子就提示
    m_discoveryTimer = new QTimer(this);
    m_discoveryTimer->setSingleShot(true);
    connect(m_discoveryTimer, &QTimer::timeout, this, &MainWindow::slot_discoveryTimeout);
    m_discoveryTimer->start(5000);

    // 文件夹监控: 每秒检查一次新文件 (等文件写完再处理)
    m_folderWatcher = new QFileSystemWatcher(this);
    connect(m_folderWatcher, &QFileSystemWatcher::directoryChanged,
            this, &MainWindow::slot_watchFolderChanged);
    m_watchTimer = new QTimer(this);
    m_watchTimer->setInterval(1000);
    connect(m_watchTimer, &QTimer::timeout, this, &MainWindow::slot_checkNewFiles);
}

// ==================== 应用模式 ====================
void MainWindow::applyMode(AppMode mode)
{
    m_currentMode = mode;

    QString roleTag = (m_currentRole == ROLE_ADMIN)
                          ? (m_isZh ? "[管理员]" : "[Admin]")
                          : (m_isZh ? "[操作员]" : "[Operator]");
    QString titleBase = m_isZh ? "基于边缘AI的晶圆缺陷检测上位机软件 V1.0"
                               : "EdgeAI Wafer Defect Inspection Host Software V1.0";
    QString modeTag = (mode == MODE_PRODUCTION)
                          ? (m_isZh ? " [产线直解]" : " [Production]")
                          : (m_isZh ? " [研发调试]" : " [R&D]");
    setWindowTitle(roleTag + " " + titleBase + modeTag);

    if (mode == MODE_PRODUCTION) {
        // 先隐藏产线面板并清空geometry, 再隐藏所有研发界面
        if (m_prodWidget) {
            m_prodWidget->hide();
            m_prodWidget->setGeometry(0, 0, 0, 0);
        }
        for (QObject *obj : centralWidget()->children()) {
            QWidget *w = qobject_cast<QWidget*>(obj);
            if (w && w != m_prodWidget) w->hide();
        }
        if (m_prodWidget) {
            m_prodWidget->setGeometry(centralWidget()->rect());
            m_prodWidget->show();
            m_prodWidget->raise();
            m_prodWidget->update();
        }
        // 产线模式隐藏整个状态栏
        statusBar()->hide();
        // 产线模式隐藏研发相关菜单项 (单文件/批量/导出CSV)
        if (m_actOpenSingle) m_actOpenSingle->setVisible(false);
        if (m_actBatch) m_actBatch->setVisible(false);
        if (m_actExportCsv) m_actExportCsv->setVisible(false);

        // 更新产线面板状态
        if (m_prodStatusLabel) {
            if (!m_discoveredIp.isEmpty()) {
                m_prodStatusLabel->setText(m_isZh ? QString("已连接板子: %1").arg(m_discoveredIp)
                                                  : QString("Board connected: %1").arg(m_discoveredIp));
                m_prodStatusLabel->setStyleSheet("font-size:14px;color:#4caf50;padding:8px;");
                if (m_prodRetryBtn) m_prodRetryBtn->hide();
            } else {
                m_prodStatusLabel->setText(m_isZh ? "未发现板子，请检查板子是否上电且在同一网段"
                                                  : "No board found. Check power and network.");
                m_prodStatusLabel->setStyleSheet("font-size:14px;color:#e74c3c;padding:8px;");
                if (m_prodRetryBtn) m_prodRetryBtn->show();
            }
        }
    } else {
        // 研发模式: 先隐藏产线面板并清空geometry置底, 再显示所有研发界面
        if (m_prodWidget) {
            m_prodWidget->hide();
            m_prodWidget->setGeometry(0, 0, 0, 0);
            m_prodWidget->lower();
        }
        for (QObject *obj : centralWidget()->children()) {
            QWidget *w = qobject_cast<QWidget*>(obj);
            if (w) w->show();
        }
        centralWidget()->update();
        statusBar()->show();
        // 恢复研发菜单项
        if (m_actOpenSingle) m_actOpenSingle->setVisible(true);
        if (m_actBatch) m_actBatch->setVisible(true);
        if (m_actExportCsv) m_actExportCsv->setVisible(true);
        // 研发模式: 隐藏"导出并下发", 保留网页看板
        if (m_exportBtn) m_exportBtn->hide();
        if (m_comboBackend) m_comboBackend->show();
        if (m_deviceBtn) m_deviceBtn->show();
        if (m_webBtn) m_webBtn->show();
        if (m_discoveryLabel) m_discoveryLabel->show();
        // 确保单图热力图状态与复选框一致
        if (ui->cbShowHeatSingle) ui->labelHeatSingle->setVisible(ui->cbShowHeatSingle->isChecked());
    }
}

// ==================== 切换模式 ====================
void MainWindow::slot_switchMode()
{
    AppMode newMode = showModeSelectDialog();
    if (newMode != m_currentMode) {
        applyMode(newMode);
        saveConfig();
        appendLog(m_isZh ? QString("[模式] 已切换到: %1").arg(newMode == MODE_PRODUCTION ? "产线直解" : "研发调试")
                         : QString("[MODE] Switched to: %1").arg(newMode == MODE_PRODUCTION ? "Production" : "R&D"));
    }
}

// ==================== 产线模式语言切换 ====================
void MainWindow::slot_prodToggleLang()
{
    m_isZh = !m_isZh;
    refreshUiText();
    // 更新产线面板所有文字
    if (m_prodLangBtn) m_prodLangBtn->setText(m_isZh ? "EN" : "中");
    if (m_prodTitleLabel) m_prodTitleLabel->setText(m_isZh ? "产线直解模式" : "Production Direct Mode");
    if (m_prodDescLabel) m_prodDescLabel->setText(
            m_isZh ? "选择晶圆数据文件（.bin / .png / .stdf），自动转换并上传到板子\n板子自动推理，结果实时推送到网页看板"
                   : "Select wafer data files (.bin / .png / .stdf)\nAuto convert & upload to board, results on web dashboard");
    if (m_prodUploadBtn) m_prodUploadBtn->setText(m_isZh ? "选择文件并下发" : "Select & Upload Files");
    if (m_prodWebBtn) m_prodWebBtn->setText(m_isZh ? "打开网页看板" : "Open Web Dashboard");
    if (m_prodLogLabel) m_prodLogLabel->setText(m_isZh ? "运行日志" : "Log");
    if (m_prodRetryBtn) m_prodRetryBtn->setText(m_isZh ? "重新发现" : "Retry");
    if (m_prodWatchBtn) m_prodWatchBtn->setText(m_isZh ? "设置监控文件夹" : "Set Watch Folder");
    if (m_prodWatchLabel && m_watchFolder.isEmpty()) {
        m_prodWatchLabel->setText(m_isZh ? "未设置监控文件夹" : "No watch folder set");
    }
    // 更新菜单
    if (m_actSwitchMode) m_actSwitchMode->setText(m_isZh ? "切换模式..." : "Switch Mode...");
    if (m_actExit) m_actExit->setText(m_isZh ? "退出" : "Exit");
    if (m_menuFile) m_menuFile->setTitle(m_isZh ? "文件(&F)" : "File(&F)");
    if (m_menuView) m_menuView->setTitle(m_isZh ? "视图(&V)" : "View(&V)");
    if (m_menuHelp) m_menuHelp->setTitle(m_isZh ? "帮助(&H)" : "Help(&H)");
    // 更新窗口标题
    applyMode(m_currentMode);
}

// ==================== 板子发现超时 ====================
void MainWindow::slot_discoveryTimeout()
{
    if (m_discoveredIp.isEmpty() && m_currentMode == MODE_PRODUCTION) {
        if (m_prodStatusLabel) {
            m_prodStatusLabel->setText(m_isZh ? "未发现板子，请检查板子是否上电且在同一网段"
                                              : "No board found. Check power and network.");
            m_prodStatusLabel->setStyleSheet("font-size:14px;color:#e74c3c;padding:8px;");
        }
        if (m_prodRetryBtn) m_prodRetryBtn->show();
        if (m_prodLog) m_prodLog->append(m_isZh ? "[警告] 5秒内未发现板子" : "[WARN] No board in 5s");
    }
}

// ==================== 设置监控文件夹 ====================
void MainWindow::slot_setWatchFolder()
{
    QString folder = QFileDialog::getExistingDirectory(this,
                                                       m_isZh ? "选择产线数据文件夹" : "Select Production Data Folder",
                                                       m_watchFolder.isEmpty() ? QDir::homePath() : m_watchFolder);
    if (folder.isEmpty()) return;

    // 移除旧监控
    if (!m_watchFolder.isEmpty() && m_folderWatcher->directories().contains(m_watchFolder)) {
        m_folderWatcher->removePath(m_watchFolder);
    }

    m_watchFolder = folder;
    m_watchKnownFiles.clear();
    m_watchPendingFiles.clear();

    // 扫描已有文件, 标记为已知 (不重复处理)
    QDir dir(folder);
    QStringList filters = {"*.bin", "*.png", "*.stdf"};
    for (const QString& f : dir.entryList(filters, QDir::Files)) {
        m_watchKnownFiles.insert(f);
    }

    m_folderWatcher->addPath(folder);
    m_watchTimer->start();
    saveConfig();

    if (m_prodWatchLabel) {
        m_prodWatchLabel->setText(m_isZh ? QString("监控中: %1").arg(folder)
                                         : QString("Watching: %1").arg(folder));
        m_prodWatchLabel->setStyleSheet("font-size:12px;color:#4caf50;");
    }
    if (m_prodLog) m_prodLog->append(m_isZh ? QString("[监控] 开始监控: %1 (已有 %2 个文件跳过)")
                                       .arg(folder).arg(m_watchKnownFiles.size())
                                 : QString("[WATCH] Watching: %1 (skipped %2 existing)")
                                       .arg(folder).arg(m_watchKnownFiles.size()));
}

// ==================== 文件夹变化通知 ====================
void MainWindow::slot_watchFolderChanged(const QString &path)
{
    Q_UNUSED(path);
    // 不立即处理, 等下一次 timer tick 检查文件大小是否稳定
    // (STDF可能还在写入中)
}

// ==================== 每秒检查新文件 (等文件写完) ====================
void MainWindow::slot_checkNewFiles()
{
    if (m_watchFolder.isEmpty() || !QDir(m_watchFolder).exists()) return;

    QDir dir(m_watchFolder);
    QStringList filters = {"*.bin", "*.png", "*.stdf"};
    QStringList files = dir.entryList(filters, QDir::Files);

    for (const QString& fname : files) {
        if (m_watchKnownFiles.contains(fname)) continue;

        QString fpath = dir.absoluteFilePath(fname);
        qint64 fsize = QFile(fpath).size();

        if (m_watchPendingFiles.contains(fname)) {
            // 已有记录, 检查大小是否稳定
            if (m_watchPendingFiles[fname] == fsize && fsize > 0) {
                // 大小稳定, 处理这个文件
                m_watchPendingFiles.remove(fname);
                m_watchKnownFiles.insert(fname);
                processWatchFile(fpath);
            } else {
                // 还在写入, 更新大小
                m_watchPendingFiles[fname] = fsize;
            }
        } else {
            // 新文件, 加入待处理
            m_watchPendingFiles[fname] = fsize;
            if (m_prodLog) m_prodLog->append(m_isZh ? QString("[新文件] %1 (等待写入完成...)").arg(fname)
                                         : QString("[NEW] %1 (waiting for write...)").arg(fname));
        }
    }
}

// ==================== 处理监控到的新文件 (转换+上传) ====================
void MainWindow::processWatchFile(const QString &fpath)
{
    QFileInfo fi(fpath);
    if (m_prodLog) m_prodLog->append(m_isZh ? QString("[处理] %1").arg(fi.fileName())
                                 : QString("[PROCESS] %1").arg(fi.fileName()));

    // 获取板子IP
    QString ip = m_discoveredIp;
    if (ip.isEmpty()) ip = m_devicePanel->getIp();
    if (ip.isEmpty()) {
        if (m_prodLog) m_prodLog->append(m_isZh ? "[失败] 未发现板子IP" : "[FAIL] No board IP");
        return;
    }

    // C++内部转换
    QByteArray edgeBin = WaferConverter::convertAuto(fpath);
    if (edgeBin.size() != WaferConverter::EDGE_BIN_SIZE) {
        if (m_prodLog) m_prodLog->append(m_isZh ? QString("[失败] 转换失败: %1").arg(fi.fileName())
                                     : QString("[FAIL] Convert: %1").arg(fi.fileName()));
        return;
    }

    // HTTP POST到板子
    QString upName = QString("%1_edge.bin").arg(fi.completeBaseName());
    QUrl url(QString("http://%1:8081/api/upload?name=%2").arg(ip, upName));
    QNetworkRequest req(url);
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/octet-stream");
    req.setHeader(QNetworkRequest::ContentLengthHeader, edgeBin.size());

    QNetworkAccessManager *nam = new QNetworkAccessManager(this);
    QNetworkReply *reply = nam->post(req, edgeBin);
    QEventLoop loop;
    connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    loop.exec();

    if (reply->error() == QNetworkReply::NoError) {
        if (m_prodLog) m_prodLog->append(m_isZh ? QString("[下发] %1 OK (12352B)").arg(upName)
                                     : QString("[UPLOAD] %1 OK (12352B)").arg(upName));
    } else {
        if (m_prodLog) m_prodLog->append(m_isZh ? QString("[失败] %1: %2").arg(upName, reply->errorString())
                                     : QString("[FAIL] %1: %2").arg(upName, reply->errorString()));
    }
    reply->deleteLater();
    nam->deleteLater();
}
