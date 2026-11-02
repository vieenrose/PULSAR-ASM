#pragma once
#include "blob.h"
void reuse_tfmamba_block(const reuse_blob_t *blob, int idx,
                         const float *x, float *out, int B, int T, int F);
