#pragma once
#include "blob.h"

void reuse_forward(const reuse_blob_t *blob, const float *mag, const float *pha,
                   int B, int F, int T,
                   float *out_mag, float *out_pha);
