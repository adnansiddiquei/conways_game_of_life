#include "array2d.h"

#ifndef AS3438_CONWAY_H
#define AS3438_CONWAY_H

// TODO: comment
namespace conway {
class ConwaysArray2DWithHalo : public array2d::Array2DWithHalo<int> {
   public:
    ConwaysArray2DWithHalo(int n_rows, int n_cols);

    void fill_randomly(float probability, int random_seed);
};
}  // namespace conway

#endif  // AS3438_CONWAY_H
