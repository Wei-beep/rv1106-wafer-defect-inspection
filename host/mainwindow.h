#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <QFileDialog>
#include <QMessageBox>
#include <QTextEdit>
#include <QLabel>
#include <QCheckBox>
#include <QPushButton>
#include <QComboBox>
#include <QTableWidget>
#include <QKeyEvent>
#include <QEvent>
#include <QDir>
#include <QShortcut>
#include <QProgressBar>
#include <QMenu>
#include <QDateTime>
#include <QSettings>
#include <QInputDialog>
#include <QLineEdit>
#include <QDesktopServices>
#include <QUrl>
#include <vector>
#include <array>
#include <QMap>
#include <QSet>
#include <QPair>
#include <QStringList>
#include <QTcpSocket>          // 新增
#include <opencv2/opencv.hpp>
#include "inferengine.h"
#include "devicepanel.h"        // 新增
#include "waferconverter.h"
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QEventLoop>
#include <QTextEdit>
#include <QTimer>
#include <QUdpSocket>
#include <QHostAddress>
#include <QFileSystemWatcher>
#include <QSet>
#include <QHash>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

enum UserRole {
    ROLE_OPERATOR = 0,
    ROLE_ADMIN = 1
};
struct InferenceResult
{
    QString fileName;
    QString fullPath;
    QDateTime importTime;
    QString sourceType;
    int classId;
    QString className;
    float confidence;
    int secondClassId;
    QString secondClassName;
    float secondConfidence;
    bool isLowConfidence;
    bool isNormal;
    int totalDie;
    int passDie;
    int failDie;
    float yieldRate;
    float defectDensity;
    float cx;
    float cy;
    float compactness;
    float aspectRatio;
    std::array<float, 8> probs;
};

enum AppMode {
    MODE_RND = 0,         // 研发调试模式
    MODE_PRODUCTION = 1   // 产线直解模式
};

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
protected:
    bool eventFilter(QObject *obj, QEvent *event) override;
    void closeEvent(QCloseEvent *event) override;
private slots:
    void slot_openSingleFile();
    void slot_batchInfer();
    void slot_exportCsv();
    void slot_ttaToggled(bool checked);
    void slot_switchLang(int idx);
    void slot_tableRowClicked(int row, int col);
    void slot_tableRowDoubleClicked(int row, int col);
    void slot_prevWafer();
    void slot_nextWafer();
    void slot_deleteSelected();
    void slot_filterChanged();
    void slot_toggleGlobalHeat();
    void slot_toggleSingleHeat(bool);
    void slot_switchRole();

    // ========== 新增槽函数 ==========
    void slot_switchBackend(int idx);
    void slot_openDeviceDialog();
    void slot_about();
    void slot_openWebDashboard();
    void slot_exportAndUpload();
    void slot_udpDiscovery();
    void slot_switchMode();
    void slot_prodToggleLang();
    void slot_discoveryTimeout();
    void slot_setWatchFolder();
    void slot_watchFolderChanged(const QString &path);
    void slot_checkNewFiles();
    void processWatchFile(const QString &fpath);

private:
    AppMode showModeSelectDialog();
    void applyMode(AppMode mode);
    void buildProductionPanel();

private:
    Ui::MainWindow *ui;
    InferEngine m_engine;
    bool m_useTTA = false;
    bool m_isZh = true;
    std::vector<InferenceResult> m_allResults;
    int m_currentIndex = -1;
    bool m_showGlobalHeat = false;
    bool m_showSingleHeat = false;
    cv::Mat m_globalHeatMap;
    QMap<QString, QVector<int>> m_batchGroups;
    QSet<QString> m_expandedBatches;
    QStringList m_batchOrder;
    QVector<QPair<bool, int>> m_displayRows;
    QStringList m_batchReports;   // 存储每个批次汇总文本（用于日志与CSV）
    QString m_filterType = "All";
    QString m_filterClass = "All";
    bool m_filterOnlyLow = false;
    QVector<int> m_filteredData;
    QSettings *m_config;
    UserRole m_currentRole;
    float m_lowConfThreshold;
    float m_normalConfThreshold;
    QString m_modelPath;
    QString m_exportDir;
    QString m_adminPassword;
    bool m_hasBreakpoint = false;
    QString m_breakpointFolder;
    int m_breakpointProcessed = 0;
    QShortcut *m_shortcutPrev;
    QShortcut *m_shortcutNext;
    QProgressBar *m_progressBar;

    // ========== 新增成员 ==========
    DevicePanel *m_devicePanel;
    QComboBox   *m_comboBackend;
    QLabel      *m_backendLabel;
    QPushButton *m_deviceBtn;
    QPushButton *m_webBtn;
    QPushButton *m_exportBtn;
    QUdpSocket *m_udpSocket;
    QLabel      *m_discoveryLabel;
    QString      m_discoveredIp;
    AppMode      m_currentMode = MODE_RND;
    QWidget     *m_prodWidget = nullptr;
    QLabel      *m_prodStatusLabel = nullptr;
    QLabel      *m_prodTitleLabel = nullptr;
    QLabel      *m_prodDescLabel = nullptr;
    QLabel      *m_prodLogLabel = nullptr;
    QTextEdit   *m_prodLog = nullptr;
    QPushButton *m_prodUploadBtn = nullptr;
    QPushButton *m_prodWebBtn = nullptr;
    QPushButton *m_prodLangBtn = nullptr;
    QPushButton *m_prodRetryBtn = nullptr;
    QPushButton *m_prodWatchBtn = nullptr;
    QLabel      *m_prodWatchLabel = nullptr;
    QTimer      *m_discoveryTimer = nullptr;
    QTimer      *m_watchTimer = nullptr;
    QFileSystemWatcher *m_folderWatcher = nullptr;
    QString      m_watchFolder;
    QSet<QString> m_watchKnownFiles;
    QHash<QString, qint64> m_watchPendingFiles;
    QAction     *m_actSwitchMode = nullptr;
    bool         m_useEdgeNpu = false;

    QMenu *m_menuFile;
    QMenu *m_menuView;
    QMenu *m_menuHelp;
    QAction *m_actOpenSingle;
    QAction *m_actBatch;
    QAction *m_actExportCsv;
    QAction *m_actExit;
    QAction *m_actGlobalHeat;
    QAction *m_actSingleHeat;
    QAction *m_actAbout;

    cv::Mat loadWaferFile(const QString& filePath);
    QPixmap cvMatToPixmap(const cv::Mat& mat);
    void appendLog(const QString& msg);
    void appendAuditLog(const QString& action, const QString& detail = "");
    void refreshUiText();
    void applyRolePermission();
    void postProcessResult(InferenceResult& r, const cv::Mat& wafer);
    void getTop2(const std::array<float, 8>& probs,
                 int& top1Id, float& top1Conf,
                 int& top2Id, float& top2Conf);
    void buildBatchGroups();
    void rebuildFilter();
    void buildDisplayRows();
    void fillResultTable();
    QString getProcessSuggestion(int classId);
    void calcWaferStats(const cv::Mat& wafer,
                        int& totalDie, int& passDie, int& failDie,
                        float& yieldRate, float& defectDensity,
                        float& cx, float& cy, float& compactness, float& aspectRatio);
    void showWaferByIndex(int globalIndex);
    void updateNavLabel();
    void buildGlobalHeatMap();
    QPixmap heatMapToPixmap(const cv::Mat& heatMap);
    QPixmap singleWaferHeatMap(const cv::Mat& wafer);
    bool checkBreakpointTask();
    void saveBreakpoint(const QString& folder, int processed);
    void clearBreakpoint();
    void loadConfig();
    void saveConfig();

    // ========== 新增函数 ==========
    void buildMenuBar();
    bool edgeInfer(const cv::Mat& wafer,
                   int& classId, float& confidence,
                   std::array<float, 8>& probs);

    static constexpr float DEFAULT_CONF_THRESHOLD = 0.70f;
    static constexpr float DEFAULT_NORMAL_THRESHOLD = 0.50f;
};
#endif
