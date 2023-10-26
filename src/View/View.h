#ifndef CPP7_MLP_SRC_VIEW_VIEW_H
#define CPP7_MLP_SRC_VIEW_VIEW_H

#include <QFile>
#include <QFileDialog>
#include <QMainWindow>
#include <QResizeEvent>
#include <QThread>
#include <QTimer>
#include <QWidget>
#include <iostream>

#include "../Controller/Controller.h"
#include "Paintscene.h"
#include "qcustomplot_lib/qcustomplot.h"
#include "ui_View.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class View;
}
QT_END_NAMESPACE

namespace s21 {

class View : public QMainWindow {
  Q_OBJECT

 public:
  explicit View(Controller *c, QWidget *parent = 0);
  ~View();

 private:
  Ui::View *ui_;
  Controller *controller_;

  QTimer *timer_;
  PaintScene *scene_;
  QGraphicsScene *loadScene_;

  std::vector<double> errorData_ = {};
  std::vector<double> accData_ = {};
  std::vector<double> epochData_ = {};

  void NewGraph();
  void ResizeEvent(QResizeEvent *event);
  void GraphCustomPlot(int maxEpoch);
  void GraphRecalcLine(int index, QPen &blue_dot_pen, int width);
  void ButtonCleanGraphicsViewClicked();
  void ButtonTrainClicked();
  void ButtonChooseTrainingClicked();
  void ButtonChooseTestingClicked();
  std::string FileDialogProcess(std::string action);
  void ButtonLoadWeightClicked();
  void ButtonSaveWeightClicked();
  void ButtonGuessClicked();
  void ButtonLoadPicClicked();
  void ButtonDelPicClicked();
  std::vector<float> CreateGuestSymbolArray(QImage *img);
  void ButtonCrossValidationClicked();
  void ButtonNumberOfGroupsClicked();
  void ShowGuessedLetter(std::string symbol);
  void UnlockButtons(bool state);
  void PrintSymbolArray(std::vector<float> guessSymbolArray);

 private slots:
  void SlotTimer();
};

}  // namespace s21

#endif  // CPP7_MLP_SRC_VIEW_VIEW_H
