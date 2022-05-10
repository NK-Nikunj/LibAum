// Copyright (C) 2022 Nikunj Gupta
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
//  Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
// more details.
//
// You should have received a copy of the GNU General Public License along
// with this program. If not, see <https://www.gnu.org/licenses/>.

#include <aum/aum.hpp>
#include <aum/ml_models/svm.hpp>

#include "ml.decl.h"

class Main : public CBase_Main
{
public:
    Main(CkArgMsg* msg)
    {
        thisProxy.benchmark();
    }

    void benchmark()
    {
        double start = CkWallTimer();

        // Initialized condition
        aum::matrix X_train(10000, 100, 1.);
        aum::vector y_train(10000, 1.);

        aum::matrix X_test(1000, 100);

        double lr = 0.001;
        int iters = 1E3;

        ml::svm logit{lr, iters};

        logit.train(X_train, y_train);
        aum::vector y_pred = logit.predict(X_test);

        aum::wait_and_exit(y_pred, start);
    }
};

#include "ml.def.h"
