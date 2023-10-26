#include <QApplication>

#include "Controller/Controller.h"
#include "Model/Model.h"
#include "View/View.h"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  std::locale::global(std::locale("C"));
  QLocale curLocale(QLocale("C"));
  QLocale::setDefault(curLocale);

  s21::Model model;
  s21::Controller controller(&model);

  s21::View view(&controller);
  view.show();
  return a.exec();
}
