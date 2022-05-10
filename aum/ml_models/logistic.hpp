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

#pragma once

#include <aum/aum.hpp>

namespace ml {
    class logistic_regression
    {
    public:
        explicit logistic_regression(double lr, int iters)
          : lr_(lr)
          , iters_(iters)
          , w_()
        {
        }

        explicit logistic_regression(aum::scalar lr, int iters)
          : lr_(lr.get())
          , iters_(iters)
          , w_()
        {
        }

        void train(aum::matrix const& X_train, aum::vector const& y_train)
        {
            w_.init(X_train.cols(), 0.);

            for (int i = 0; i != iters_; ++i)
            {
                aum::vector z = aum::dot(X_train, w_);
                z.sigmoid();
                aum::vector grads = aum::dot(z, X_train) / y_train.size();
                w_ = aum::blas::axpy(-lr_, grads, w_);
            }
        }

        aum::vector predict(aum::matrix const& X_test)
        {
            aum::vector y_pred = aum::dot(X_test, w_);
            y_pred.sigmoid();

            return y_pred;
        }

    private:
        double lr_;
        int iters_;
        aum::vector w_;
    };

}    // namespace ml
