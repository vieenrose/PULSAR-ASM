// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "wav_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int wav_read_f32(const char *path, float **data, int *nch, int *sr, int *nframes) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char id[4]; fread(id,1,4,f);
    uint32_t sz; fread(&sz,1,4,f);
    char type[4]; fread(type,1,4,f);
    int sample_rate = 0;
    int num_channels = 1;
    while (1) {
        char cid[4]; if (fread(cid,1,4,f)!=4) break;
        uint32_t csz; fread(&csz,1,4,f);
        if (memcmp(cid,"fmt ",4)==0) {
            uint16_t audio_format; fread(&audio_format,1,2,f);
            uint16_t num_channels16; fread(&num_channels16,1,2,f);
            uint32_t sample_rate32; fread(&sample_rate32,1,4,f);
            uint32_t byte_rate; fread(&byte_rate,1,4,f);
            uint16_t block_align; fread(&block_align,1,2,f);
            uint16_t bits_per_sample; fread(&bits_per_sample,1,2,f);
            if (csz > 16) fseek(f, csz-16, SEEK_CUR);
            *nch = num_channels16; *sr = sample_rate32;
            num_channels = num_channels16;
            sample_rate = sample_rate32;
        } else if (memcmp(cid,"data",4)==0) {
            int n = csz / 2;
            short *tmp = (short *)malloc(csz);
            fread(tmp,1,csz,f);
            *nframes = n / (num_channels ? num_channels : 1);
            *data = (float *)malloc((*nframes)*(num_channels)*sizeof(float));
            for (int i=0;i<n;i++) (*data)[i] = tmp[i] / 32768.0f;
            free(tmp);
            fclose(f);
            return 0;
        } else {
            fseek(f, csz + (csz%2), SEEK_CUR);
        }
    }
    fclose(f);
    return -2;
}
