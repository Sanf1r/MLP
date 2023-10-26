#include "View.h"

namespace s21 {

View::View(Controller *c, QWidget *parent)
    : QMainWindow(parent), ui_(new Ui::View), controller_(c) {
  ui_->setupUi(this);

  connect(ui_->buttonChooseTraining, &QPushButton::clicked, this,
          &View::ButtonChooseTrainingClicked);
  connect(ui_->buttonChooseTesting, &QPushButton::clicked, this,
          &View::ButtonChooseTestingClicked);

  connect(ui_->buttonTrain, &QPushButton::clicked, this,
          &View::ButtonTrainClicked);

  connect(ui_->buttonSaveWeight, &QPushButton::clicked, this,
          &View::ButtonSaveWeightClicked);

  connect(ui_->buttonLoadWeight, &QPushButton::clicked, this,
          &View::ButtonLoadWeightClicked);

  connect(ui_->buttonLoadPic, &QPushButton::clicked, this,
          &View::ButtonLoadPicClicked);

  connect(ui_->buttonDelPic, &QPushButton::clicked, this,
          &View::ButtonDelPicClicked);

  connect(ui_->buttonCleanGraphicsView, &QPushButton::clicked, this,
          &View::ButtonCleanGraphicsViewClicked);

  connect(ui_->buttonGuess, &QPushButton::clicked, this,
          &View::ButtonGuessClicked);

  connect(ui_->checkBoxCrossValidationMode, &QCheckBox::clicked, this,
          &View::ButtonCrossValidationClicked);

  connect(ui_->spinboxNumberOfGroups, &QSpinBox::valueChanged, this,
          &View::ButtonNumberOfGroupsClicked);

  scene_ = new PaintScene();
  timer_ = new QTimer();

  connect(timer_, &QTimer::timeout, this, &View::SlotTimer);

  NewGraph();
}

View::~View() {
  delete ui_;
  delete scene_;
  delete timer_;
}

void View::NewGraph() {
  scene_->clear();
  timer_->stop();

  ui_->graphicsView->setScene(NULL);
  ui_->graphicsView->setScene(scene_);

  timer_->start(100);
}

/**
 * @brief Auxiliary method for preparing the scene
 *
 */
void View::SlotTimer() {
  timer_->stop();
  scene_->setSceneRect(0, 0, ui_->graphicsView->width() - 20,
                       ui_->graphicsView->height() - 20);
}

/**
 * @brief For draw a graph of data changes by epoch
 *
 * @param maxEpoch
 */
void View::GraphCustomPlot(int maxEpoch) {
  QVector<double> q_e = QVector<double>(controller_->GetError().begin(),
                                        controller_->GetError().end());
  QVector<double> q_a = QVector<double>(controller_->GetAccur().begin(),
                                        controller_->GetAccur().end());
  QVector<double> q_ch = QVector<double>(controller_->getEpoch().begin(),
                                         controller_->getEpoch().end());

  ui_->widget->addGraph(ui_->widget->xAxis, ui_->widget->yAxis);
  ui_->widget->graph(0)->setData(q_ch, q_e);
  ui_->widget->graph(0)->setPen(QPen(Qt::red));
  ui_->widget->graph(0)->setName("Mean Square Error");

  ui_->widget->addGraph(ui_->widget->xAxis2, ui_->widget->yAxis2);
  ui_->widget->graph(1)->setData(q_ch, q_a);
  ui_->widget->graph(1)->setName("Train Accuracy");

  ui_->widget->xAxis->setLabel("Epoch");
  ui_->widget->yAxis->setLabel("Errors");
  ui_->widget->yAxis2->setLabel("Accuracy");

  ui_->widget->xAxis2->setVisible(true);
  ui_->widget->yAxis2->setVisible(true);

  int start = (ui_->checkBoxCrossValidationMode->checkState()) ? 1 : 0;
  ui_->widget->xAxis->setRange(start, maxEpoch);
  ui_->widget->yAxis->setRange(0, q_e[0] + 0.05);
  ui_->widget->xAxis2->setRange(start, maxEpoch);
  ui_->widget->yAxis2->setRange(0, 1);

  ui_->widget->replot();
}

/**
 * @brief Setting up the line parameters for the graph
 *
 * @param index
 * @param blue_dot_pen
 * @param width
 */
void View::GraphRecalcLine(int index, QPen &blue_dot_pen, int width) {
  blue_dot_pen.setColor("Red");
  switch (index) {
    case 0:
      blue_dot_pen.setStyle(Qt::SolidLine);
      break;
    case 1:
      blue_dot_pen.setStyle(Qt::DotLine);
      break;
    case 2:
      blue_dot_pen.setStyle(Qt::DashLine);
      break;
    case 3:
      blue_dot_pen.setStyle(Qt::DashDotLine);
      break;
    case 4:
      blue_dot_pen.setStyle(Qt::DashDotDotLine);
      break;
    default:
      blue_dot_pen.setStyle(Qt::SolidLine);
      break;
  }
  blue_dot_pen.setWidthF(width);
}

/**
 * @brief Auxiliary method for changing the size of the scene
 *
 * @param event
 */
void View::ResizeEvent(QResizeEvent *event) {
  timer_->start(100);
  QWidget::resizeEvent(event);
}

/**
 * @brief Cleans the scene for drawing
 *
 */
void View::ButtonCleanGraphicsViewClicked() {
  ui_->graphicsView->setScene(NULL);
  NewGraph();
  ShowGuessedLetter("");
}

/**
 * @brief Launches the process of training the neural network
 *
 */
void View::ButtonTrainClicked() {
  std::string pathTrain = ui_->lineeditFileNameTraining->text().toStdString();
  std::string pathTest = ui_->lineeditFileNameTest->text().toStdString();
  if (!pathTrain.empty() && !pathTest.empty()) {
    UnlockButtons(false);
    int cEpoch = 1;

    controller_->SetSettingsData(
        ui_->comboboxLayersNumber->currentText().toInt(),
        ui_->spinboxNumberOfEpoch->value(),
        ui_->doublespinboxSampleSize->value(),
        ui_->checkBoxCrossValidationMode->checkState(),
        ui_->spinboxNumberOfGroups->value(),
        ui_->radiobuttonMatrixImpl->isChecked(), pathTrain, pathTest);

    auto t1 = std::chrono::high_resolution_clock::now();
    if (ui_->radiobuttonMatrixImpl->isChecked()) {
      if (ui_->checkBoxCrossValidationMode->checkState()) {
        controller_->PrepareCrossTrain();
        while (cEpoch <= ui_->spinboxNumberOfEpoch->value()) {
          controller_->RunCrossTrain(cEpoch++);
        }
      } else {
        controller_->PrepareTrain();
        while (cEpoch <= ui_->spinboxNumberOfEpoch->value()) {
          controller_->RunTrain(cEpoch++);
        }
      }
    } else {
      if (ui_->checkBoxCrossValidationMode->checkState()) {
        controller_->PrepareGraphCrossTrain();
        while (cEpoch <= ui_->spinboxNumberOfEpoch->value()) {
          controller_->RunGraphCrossTrain(cEpoch++);
        }
      } else {
        controller_->PrepareGraphTrain();
        while (cEpoch <= ui_->spinboxNumberOfEpoch->value()) {
          controller_->RunGraphTrain(cEpoch++);
        }
      }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto total =
        std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    GraphCustomPlot(ui_->spinboxNumberOfEpoch->value());
    ui_->textTerminal->setText(
        QString::fromStdString(controller_->GetTerminalData()));
    ui_->textTerminal->append("End train.");
    ui_->textTerminal->append("Total experiment statistic:");
    ui_->textTerminal->append(
        "Average accuracy = " +
        QString::fromStdString(std::to_string(controller_->GetAccStat())));
    ui_->textTerminal->append(
        "Precision = " +
        QString::fromStdString(std::to_string(controller_->GetPrecStat())));
    ui_->textTerminal->append(
        "Recall = " +
        QString::fromStdString(std::to_string(controller_->GetRecStat())));
    ui_->textTerminal->append(
        "F-measure = " +
        QString::fromStdString(std::to_string(controller_->GetMeasStat())));
    ui_->textTerminal->append(
        "Total time = " + QString::fromStdString(std::to_string(total)) +
        " seconds.");
  }

  UnlockButtons(true);
}

/**
 * @brief Selecting a file for training
 *
 */
void View::ButtonChooseTrainingClicked() { FileDialogProcess("train"); }

/**
 * @brief Selecting a file for testing
 *
 */
void View::ButtonChooseTestingClicked() { FileDialogProcess("test"); }

/**
 * Auxiliary method for opening a dialog box
 */
std::string View::FileDialogProcess(std::string action) {
  QString filter = " ALL Files (*)";
  QString fileNameTmp = QFileDialog::getOpenFileName(
      this, "Open test file", QDir::currentPath(), filter);
  if (!fileNameTmp.isEmpty() && !fileNameTmp.isNull()) {
    if (action == "train") {
      ui_->lineeditFileNameTraining->setText(fileNameTmp);
    } else if (action == "test") {
      ui_->lineeditFileNameTest->setText(fileNameTmp);
    } else if (action == "weight") {
    }
  }
  return fileNameTmp.toStdString();
}

/**
 * @brief Launches the process of uploading a file with weights
 *
 */
void View::ButtonLoadWeightClicked() {
  std::string path = FileDialogProcess("weight");
  if (!path.empty()) {
    bool stateMatrix = controller_->LoadWeight(path);
    if (stateMatrix) {
      ui_->radiobuttonMatrixImpl->setChecked(true);
      ui_->radiobuttonGraphImpl->setChecked(false);
    } else {
      ui_->radiobuttonMatrixImpl->setChecked(false);
      ui_->radiobuttonGraphImpl->setChecked(true);
    }

    ui_->textTerminal->setText(
        QString::fromStdString(controller_->GetTerminalData()));
  }
}

/**
 * @brief Launches the process of saving a file with weights
 *
 */
void View::ButtonSaveWeightClicked() {
  QString filename = QFileDialog::getSaveFileName(this, tr("Save weight file"),
                                                  QDir::currentPath(),
                                                  tr("Text Files (*.csv)"));
  if (filename != "") {
    std::string path = filename.toStdString();
    controller_->SaveWeight(path);
  }
}

/**
 * @brief Start the process to recognize the symbol
 *
 */
void View::ButtonGuessClicked() {
  QImage img(28, 28, QImage::Format_ARGB32);
  img.fill(Qt::black);
  QPainter painter(&img);
  scene_->render(&painter);
  painter.end();

  std::vector<float> symbArray = CreateGuestSymbolArray(&img);
  controller_->ButtonGuessClicked(symbArray);

  ShowGuessedLetter(controller_->GetGuessedSymbol());
}

/**
 * @brief For choosing a picture as well as for recognition
 *
 */
void View::ButtonLoadPicClicked() {
  QString filter = " BMP Files (*.bmp)";
  QString fileNameTmp = QFileDialog::getOpenFileName(
      this, "Load picture...", QDir::currentPath(), filter);
  if (!fileNameTmp.isEmpty() && !fileNameTmp.isNull()) {
    ui_->graphicsviewLoadPic->setScene(NULL);
    QPixmap pic(fileNameTmp);
    if (!ui_->graphicsviewLoadPic->scene()) {
      loadScene_ = new QGraphicsScene(this);
      ui_->graphicsviewLoadPic->setScene(loadScene_);
    }
    QPixmap scaled_img =
        pic.scaled(ui_->graphicsviewLoadPic->width() - 5,
                   ui_->graphicsviewLoadPic->height() - 5, Qt::KeepAspectRatio);
    ui_->graphicsviewLoadPic->scene()->addPixmap(scaled_img);
    ui_->graphicsviewLoadPic->scene()->setSceneRect(scaled_img.rect());

    QPixmap scaledImg28 = pic.scaled(28, 28, Qt::KeepAspectRatio);
    QImage *img = new QImage;
    *img = scaledImg28.toImage();

    std::vector<float> symbArray = CreateGuestSymbolArray(img);
    controller_->ButtonGuessClicked(symbArray);
    ShowGuessedLetter(controller_->GetGuessedSymbol());
  }
}

/**
 * @brief To clean up a window with a picture
 *
 */
void View::ButtonDelPicClicked() {
  if (ui_->graphicsviewLoadPic->scene()) {
    ui_->graphicsviewLoadPic->setScene(NULL);
  }
}

/**
 * @brief Creates an array from a particular picture of the symbol
 *
 * @param img
 * @return std::vector<float>
 */
std::vector<float> View::CreateGuestSymbolArray(QImage *img) {
  std::vector<float> guessSymbolArray;
  if (false == img->isNull()) {
    for (int row = 0; row < img->height(); ++row) {
      for (int col = 0; col < img->width(); ++col) {
        QColor clrCurrent(img->pixel(row, col));
        guessSymbolArray.push_back(
            (clrCurrent.red() + clrCurrent.green() + clrCurrent.blue()) / 3.0 /
            255.0);
      }
    }
  }
  return guessSymbolArray;
}

/**
 * @brief For lock/unlock the number of epoch fields when changing
 * cross-validation mode
 *
 */
void View::ButtonCrossValidationClicked() {
  if (ui_->checkBoxCrossValidationMode->isChecked()) {
    ui_->spinboxNumberOfEpoch->setEnabled(false);
    ui_->spinboxNumberOfEpoch->setValue(ui_->spinboxNumberOfGroups->value());
  } else {
    ui_->spinboxNumberOfEpoch->setEnabled(true);
  }
}

/**
 * @brief Change the field the number of epoch if the button cross-validation
 * mode is pressed
 *
 */
void View::ButtonNumberOfGroupsClicked() {
  if (ui_->checkBoxCrossValidationMode->isChecked()) {
    ui_->spinboxNumberOfEpoch->setValue(ui_->spinboxNumberOfGroups->value());
  }
}

/**
 * @brief Sets the field value with a new symbol
 *
 * @param symbol
 */
void View::ShowGuessedLetter(std::string symbol) {
  ui_->labelGuess->setText(QString::fromStdString(symbol));
}

/**
 * @brief Block/Unblock a group of buttons
 *
 * @param state
 */
void View::UnlockButtons(bool state) {
  ui_->groupBoxTrain->setEnabled(state);
  ui_->groupBoxSettings->setEnabled(state);
  ui_->groupBoxDemonstration->setEnabled(state);
}

/**
 * @brief Auxiliary method for printing a symbol as an array
 *
 * @param guessSymbolArray
 */
void View::PrintSymbolArray(std::vector<float> guessSymbolArray) {
  for (size_t i = 0; i < guessSymbolArray.size(); ++i) {
    std::cout << guessSymbolArray.at(i) << " ";
    if ((i + 1) % 28 == 0) {
      std::cout << std::endl;
    }
  }
}

}  // namespace s21
