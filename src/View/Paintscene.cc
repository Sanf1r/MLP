#include "Paintscene.h"

namespace s21 {

PaintScene::PaintScene(QObject *parent) : QGraphicsScene(parent) {}
PaintScene::~PaintScene() {}

void PaintScene::mousePressEvent(QGraphicsSceneMouseEvent *event) {
  addEllipse(event->scenePos().x() - 5, event->scenePos().y() - 5, 40, 40,
             QPen(Qt::NoPen), QBrush(Qt::white));
  previousPoint_ = event->scenePos();
}

void PaintScene::mouseMoveEvent(QGraphicsSceneMouseEvent *event) {
  addLine(previousPoint_.x(), previousPoint_.y(), event->scenePos().x(),
          event->scenePos().y(),
          QPen(Qt::white, 40, Qt::SolidLine, Qt::RoundCap));
  previousPoint_ = event->scenePos();
}

}  // namespace s21
