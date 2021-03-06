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

#include "matrices.decl.h"

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
        aum::matrix A{1000, 1000, 1.1};
        aum::matrix B{1000, 1000, 2.2};
        aum::matrix C{1000, 1000, 3.3};
        aum::matrix D{1000, 1000, 4.4};

        // Force 2 temporaries
        aum::matrix E = (A - D) - (B - C);

        // 1 temp to the left
        A = B - C - D;

        // 1 temp to the right
        B = C - (A - D);

        aum::wait_and_exit(B, start);
    }
};

#include "matrices.def.h"
