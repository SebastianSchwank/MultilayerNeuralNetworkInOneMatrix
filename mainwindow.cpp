#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    timer = new QTimer(this);

    //Initalize completly connected stacked neural net
    Cluster0 = new NeuralCluster(numInputs,numOutputs,32);
}

MainWindow::~MainWindow()
{
    delete ui;
}

vector<float> MainWindow::inputFunction(int type, int length,int periode){
    vector<float> function;
    if(type == 0){
        for(int i = 0; i < length; i++){
          function.push_back(0.5-0.5*sin(3.16*1.0*numInputs*i/(length*periode)));
        }
    }
    if(type == 1){
        int toggle = 0;
        for(int i = 0; i < length; i++){
            if((i%(periode)) == 0) toggle = 1-toggle;
            if(toggle == 0) function.push_back(1.0);
            if(toggle == 1) function.push_back(0.0);
        }
    }
    if(type == 2){
        int toggle = 0;
        for(int i = 0; i < length; i++){
            if((i%(periode)) == 0) function.push_back(1.0);
            else function.push_back(0.0);
        }
    }
    return function;
}

void MainWindow::processNet(){
        float sumErrorOver = 0.0;

        //Test pass error calculation
        currentFrequency = (currentFrequency+1)%(numInputs/2);
        int frequency = (currentFrequency+3);

        for(int k = 0; k < numLessons; k++){

            int i = k;

            //Create empty vector as output placeholder
                vector<float> emptyV;
            //Create input vector for holding the input data (Frequency is random Waveform depends on the lesson number (is mapped to output-neurons))
                vector<float> inputV = MainWindow::inputFunction(k,numInputs,frequency);

                vector<float> targetV;
                for(int j = 0; j < numOutputs; j++){
                    targetV.push_back(0.0);
                    //cout << inputV[j];
                }
                targetV[i] = 1.0;
                targetV[3+frequency] = 1.0;


                for(int i = 0; i < 32; i++){
                    Cluster0->propergate(inputV,emptyV,0.0);
                }
                vector<float> out0 = Cluster0->getActivation();

                for(int i = 0; i < numOutputs; i++){
                    sumErrorOver += (targetV[i]-out0[i+numInputs])*(targetV[i]-out0[i+numInputs]);
                }

        }

        //Test pass visualized

        vector<vector<float>> impulseResonses;
        for(int k = 0; k < numLessons; k++){

            int i = k;

            //Create empty vector as output placeholder
                vector<float> emptyV;

            //Create input vector for holding the input data (Frequency is random Waveform depends on the lesson number (is mapped to output-neurons))
                vector<float> inputV = MainWindow::inputFunction(k,numInputs,frequency);

                vector<float> targetV;
                for(int j = 0; j < numOutputs; j++){
                    targetV.push_back(0.0);
                }
                targetV[i] = 1.0;
                targetV[3+frequency] = 1.0;

                for(int i = 0; i < 32; i++){
                    Cluster0->propergate(inputV,targetV,0.0);
                }
                vector<float> out0 = Cluster0->getActivation();
                vector<float> out1 = Cluster0->getCounterActivation();

                impulseResonses.push_back(out0);
                impulseResonses.push_back(out1);

        }

        //Training pass

        for(int k = 0; k < numLessons; k++){

            int i = k;

            //Create empty vector as output placeholder
                vector<float> emptyV;

                //Create input vector for holding the input data (Frequency is random Waveform depends on the lesson number (is mapped to output-neurons))
                vector<float> inputV = MainWindow::inputFunction(k,numInputs,frequency);

                vector<float> targetV;
                for(int j = 0; j < numOutputs; j++){
                    targetV.push_back(0.0);
                }
                targetV[k] = 1.0;
                targetV[3+frequency] = 1.0;


                for(int i = 0; i < 32; i++){

                    Cluster0->propergate(inputV,targetV,0.01);
                    Cluster0->train();
                }

            }


        imageResp = new QImage(impulseResonses[0].size()*4,numLessons*4*2,QImage::Format_RGB32);
        for(int x = 0; x < numLessons*4*2; x++){
            for(int y = 0; y < impulseResonses[0].size()*4; y++){
                QColor col = QColor(128,128,128);
                float colorVal = (impulseResonses[x/4][y/4]-0.5)*2.0;
                if(colorVal > 0.0) col = QColor(255.0*abs(colorVal),0,0);
                if(colorVal < 0.0) col = QColor(0,0,255.0*abs(colorVal));
                imageResp->setPixel(y,x,col.rgb());
            }
        }




        image = new QImage(Cluster0->getActivation().size(),Cluster0->getActivation().size(),QImage::Format_RGB32);
        float maxVal = 0.0;
        for(int x = 0; x < Cluster0->getActivation().size(); x++){
            for(int y = 0; y < Cluster0->getActivation().size(); y++){
                if(abs(Cluster0->getWeights()[y][x]) > maxVal) maxVal=abs(Cluster0->getWeights()[y][x]);
            }
        }

        for(int x = 0; x < Cluster0->getActivation().size(); x++){
            for(int y = 0; y < Cluster0->getActivation().size(); y++){
                QColor col = QColor(128,128,128);

                if(Cluster0->getWeights()[y][x]/maxVal > 0.0) col = QColor(255.0*Cluster0->getWeights()[y][x]/maxVal,0,0);
                if(Cluster0->getWeights()[y][x]/maxVal < 0.0) col = QColor(0,0,-255.0*Cluster0->getWeights()[y][x]/maxVal);

                image->setPixel(x,y,col.rgb());
            }
        }

        QString Text;
        Text = QString::number(sumErrorOver/(numOutputs*numLessons)) + "\n";

       ui->textBrowser->append(Text);

           QGraphicsScene* scene = new QGraphicsScene;
           scene->addPixmap(QPixmap::fromImage(*image));


           QGraphicsScene* scene2 = new QGraphicsScene;
           scene2->addPixmap(QPixmap::fromImage(*imageResp));

           ui->graphicsView2->setScene(scene2);
           ui->graphicsView->setScene( scene );

           ui->graphicsView->show();
}

void MainWindow::on_pushButton_clicked()
{
    if(running == false){
        connect(timer, SIGNAL(timeout()), this, SLOT(processNet()));
        timer->start();
        running = !running;
    }else {
        timer->stop();
        running = !running;
    }
}
