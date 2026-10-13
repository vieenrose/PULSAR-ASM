/* peakrss: exec a child, report its exact peak RSS (wait4 ru_maxrss) to $PEAKRSS_OUT.
 * Child stdout/stderr untouched. Usage: PEAKRSS_OUT=/tmp/rss.txt peakrss prog args... */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
int main(int argc, char **argv) {
    pid_t p;
    int st;
    struct rusage ru;
    const char *o;
    FILE *f;
    if (argc < 2) return 2;
    p = fork();
    if (p == 0) { execvp(argv[1], argv + 1); _exit(127); }
    if (p < 0) return 2;
    wait4(p, &st, 0, &ru);
    o = getenv("PEAKRSS_OUT");
    if (o) {
        f = fopen(o, "a");
        if (f) { fprintf(f, "PEAKRSS_KB %ld\n", (long)ru.ru_maxrss); fclose(f); }
    }
    return WIFEXITED(st) ? WEXITSTATUS(st) : 3;
}
