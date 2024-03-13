#include <iostream>

#include "conway.h"

int main(int argc, char *argv[]) {
    for (int i = 100; i < 1000; i += 100) {
        conway::ConwaysArray2DWithHalo grid(i, i);
        grid.fill_randomly(0.5, -1);
    }
}