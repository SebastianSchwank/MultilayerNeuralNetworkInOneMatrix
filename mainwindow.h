#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPixmap>
#include <QTimer>
#include <QDebug>

#include "neuralcluster.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();
    void processNet();

private:
    vector<float> inputFunction(int type, int length,int periode);

private:
    Ui::MainWindow *ui;

    QImage *image;
    QImage *imageResp;

    bool running = false;

    int numLessons = 3;//sizeof (input)/sizeof (input[0]);

    int currentFrequency = 0;
    int numInputs = 16;//sizeof (input[0])/sizeof (input[0][0]);
    int numOutputs = 3+10;//sizeof (output[0])/sizeof (output[0][0]);
    NeuralCluster* Cluster0;
    QTimer *timer;
};

#endif // MAINWINDOW_H
