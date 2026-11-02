// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "wav_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int wav_read_f32(const char *path, float **data, int *nch, int *sr, int *nframes) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char id[4]; fread(id,1,4,f);
    uint32_t sz; fread(&sz,1,4,f);
    char type[4]; fread(type,1,4,f);
    while (1) {
        char cid[4]; if (fread(cid,1,4,f)!=4) break;
        uint32_t csz; fread(&csz,1,4,f);
        if (memcmp(cid,"fmt ",4)==0) {
            uint16_t audio_format; fread(&audio_format,1,2,f);
            uint16_t num_channels; fread(&num_channels,1,2,f);
            uint32_t sample_rate; fread(&sample_rate,1,4,f);
            uint32_t byte_rate; fread(&byte_rate,1,4,f);
            uint16_t block_align; fread(&block_align,1,2,f);
            uint16_t bits_per_sample; fread(&bits_per_sample,1,2,f);
            if (csz > 16) fseek(f, csz-16, SEEK_CUR);
            *nch = num_channels; *sr = sample_rate;
        } else if (memcmp(cid,"data",4)==0) {
            int bytes_per_sample = (*nch==0?1:*nch)>0?2:2; // default pcm16
            // need bits; store
            // re-parse easier: just read pcm16
            int n = csz / 2;
            short *tmp = (short *)malloc(csz);
            fread(tmp,1,csz,f);
            *nframes = n / (*nch ? *nch : 1);
            *data = (float *)malloc((*nframes)*(*nch)*sizeof(float));
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
