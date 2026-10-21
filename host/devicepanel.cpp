#include "devicepanel.h"

DevicePanel::DevicePanel(QWidget *parent)
    : QDialog(parent)
{
    setFixedSize(340, 175);

    m_ipEdit    = new QLineEdit("192.168.1.100", this);
    m_portEdit  = new QLineEdit("8080", this);
    m_portEdit->setValidator(new QIntValidator(1, 65535, this));

    m_ipLabel   = new QLabel("Device IP:", this);
    m_portLabel = new QLabel("Port:", this);
    m_connectBtn = new QPushButton("Connect", this);
    m_statusText = new QLabel("Disconnected", this);
    m_statusLight = new QLabel(this);
    m_statusLight->setFixedSize(16, 16);
    m_statusLight->setStyleSheet("border-radius:8px; background:#e74c3c;");

    QFormLayout *form = new QFormLayout();
    form->addRow(m_ipLabel, m_ipEdit);
    form->addRow(m_portLabel, m_portEdit);

    QHBoxLayout *statusBar = new QHBoxLayout();
    statusBar->addWidget(m_statusLight);
    statusBar->addWidget(m_statusText);
    statusBar->addStretch();
    statusBar->addWidget(m_connectBtn);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->addLayout(form);
    mainLayout->addSpacing(8);
    mainLayout->addLayout(statusBar);

    m_socket = new QTcpSocket(this);

    connect(m_connectBtn, &QPushButton::clicked, this, &DevicePanel::onConnectClicked);
    connect(m_socket, &QTcpSocket::connected,    this, &DevicePanel::onSocketConnected);
    connect(m_socket, &QTcpSocket::disconnected, this, &DevicePanel::onSocketDisconnected);
    connect(m_socket, SIGNAL(error(QAbstractSocket::SocketError)),
            this, SLOT(onSocketError(QAbstractSocket::SocketError)));
}

DevicePanel::~DevicePanel()
{
    if (m_socket->isOpen()) m_socket->close();
}

bool DevicePanel::isConnected() const
{
    return m_connected && m_socket->state() == QAbstractSocket::ConnectedState;
}

QTcpSocket* DevicePanel::socket()
{
    return m_socket;
}

void DevicePanel::setIp(const QString& ip)
{
    if (m_ipEdit) m_ipEdit->setText(ip);
}

void DevicePanel::setLanguage(bool zh)
{
    m_zh = zh;
    setWindowTitle(zh ? "边缘设备连接" : "Edge Device");
    m_ipLabel->setText(zh ? "设备 IP：" : "Device IP:");
    m_portLabel->setText(zh ? "端口号：" : "Port:");
    m_ipEdit->setToolTip(zh ? "RV1106 板子的 IP 地址" : "RV1106 board IP address");
    updateStatusUi();
}

void DevicePanel::onConnectClicked()
{
    if (m_connected) {
        m_socket->disconnectFromHost();
        return;
    }
    QString ip = m_ipEdit->text().trimmed();
    if (ip.isEmpty()) {
        QMessageBox::warning(this, m_zh?"提示":"Tip",
                             m_zh ? "请输入设备 IP 地址" : "Please enter device IP");
        return;
    }
    int port = m_portEdit->text().toInt();

    m_connectBtn->setEnabled(false);
    m_statusText->setText(m_zh ? "连接中..." : "Connecting...");
    m_socket->connectToHost(ip, port);
}

void DevicePanel::onSocketConnected()
{
    m_connected = true;
    m_connectBtn->setEnabled(true);
    updateStatusUi();
    emit connectionChanged(true);
}

void DevicePanel::onSocketDisconnected()
{
    m_connected = false;
    m_connectBtn->setEnabled(true);
    updateStatusUi();
    emit connectionChanged(false);
}

void DevicePanel::onSocketError(QAbstractSocket::SocketError err)
{
    Q_UNUSED(err);
    m_connectBtn->setEnabled(true);
    m_statusText->setText(m_zh ? "连接失败" : "Connect failed");
    QMessageBox::critical(this, m_zh?"连接错误":"Connection Error",
                          m_socket->errorString());
}

void DevicePanel::updateStatusUi()
{
    if (m_connected) {
        m_statusLight->setStyleSheet("border-radius:8px; background:#27ae60;");
        m_statusText->setText(m_zh ? "已连接" : "Connected");
        m_connectBtn->setText(m_zh ? "断开" : "Disconnect");
    } else {
        m_statusLight->setStyleSheet("border-radius:8px; background:#e74c3c;");
        m_statusText->setText(m_zh ? "未连接" : "Disconnected");
        m_connectBtn->setText(m_zh ? "连接" : "Connect");
    }
}
