#ifndef DATAH
#define DATAH

class Box {
  public:
    float left, top, right, bottom;
    Box(){}
    Box(float left, float top, float right, float bottom):
        left(left), top(top), right(right), bottom(bottom){}
};

class Detection {
  public:
    Box box;
    float score, label;
    Detection(Box const& box, float score, float label):
        box(box), score(score), label(label){}
};

#endif