#pragma once

#include "func/unary.h"
#include "func/binary.h"
#include "func/linalg.h"
#include "func/reduce.h"
#include "api/creation.h"
#include "api/tape.h"

using radann::operator+;
using radann::operator-;
using radann::operator*;
using radann::operator/;

using radann::operator""_C;
using radann::operator""_fC;
