#ifndef PAINTSCENE_H
#define PAINTSCENE_H

#include <QDebug>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QTimer>

namespace s21 {

class PaintScene : public QGraphicsScene {
  Q_OBJECT

 public:
  explicit PaintScene(QObject *parent = 0);
  ~PaintScene();

 private:
  QPointF previousPoint_;
  void mousePressEvent(QGraphicsSceneMouseEvent *event);
  void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
};

}  // namespace s21

#endif  // PAINTSCENE_H
