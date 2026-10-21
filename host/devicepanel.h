#ifndef DEVICEPANEL_H
#define DEVICEPANEL_H

#include <QDialog>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QTcpSocket>
#include <QIntValidator>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QMessageBox>

class DevicePanel : public QDialog
{
    Q_OBJECT
public:
    explicit DevicePanel(QWidget *parent = nullptr);
    ~DevicePanel();

    bool isConnected() const;
    QTcpSocket* socket();
    void setLanguage(bool zh);   // 供主窗口切换中英文
    QString getIp() const { return m_ipEdit ? m_ipEdit->text().trimmed() : QString(); }
    void setIp(const QString& ip);   // UDP自动发现后填入IP


signals:
    void connectionChanged(bool connected);

private slots:
    void onConnectClicked();
    void onSocketConnected();
    void onSocketDisconnected();
    void onSocketError(QAbstractSocket::SocketError err);

private:
    QLineEdit   *m_ipEdit;
    QLineEdit   *m_portEdit;
    QPushButton *m_connectBtn;
    QLabel      *m_ipLabel;
    QLabel      *m_portLabel;
    QLabel      *m_statusText;
    QLabel      *m_statusLight;

    QTcpSocket  *m_socket;
    bool         m_connected = false;
    bool         m_zh = true;

    void updateStatusUi();
};

#endif // DEVICEPANEL_H
