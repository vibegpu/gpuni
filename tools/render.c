#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#if defined(_WIN32)
#  include <direct.h>
#  define PK_PATH_SEP '\\'
#else
#  include <unistd.h>
#  define PK_PATH_SEP '/'
#endif

typedef struct {
  char **items;
  size_t count;
  size_t capacity;
} pk_str_list;

static void pk_die(const char *msg) {
  fprintf(stderr, "render: %s\n", msg);
  exit(1);
}

static void pk_die_errno(const char *context) {
  fprintf(stderr, "render: %s: %s\n", context, strerror(errno));
  exit(1);
}

static void *pk_xmalloc(size_t n) {
  void *p = malloc(n);
  if (!p) pk_die("out of memory");
  return p;
}

static void *pk_xrealloc(void *p, size_t n) {
  void *q = realloc(p, n);
  if (!q) pk_die("out of memory");
  return q;
}

static char *pk_xstrdup(const char *s) {
  size_t n = strlen(s) + 1;
  char *p = (char *)pk_xmalloc(n);
  memcpy(p, s, n);
  return p;
}

static void pk_str_list_push(pk_str_list *list, char *owned) {
  if (list->count == list->capacity) {
    size_t new_capacity = list->capacity ? list->capacity * 2 : 8;
    list->items = (char **)pk_xrealloc(list->items, new_capacity * sizeof(list->items[0]));
    list->capacity = new_capacity;
  }
  list->items[list->count++] = owned;
}

static int pk_str_list_contains(const pk_str_list *list, const char *s) {
  size_t i;
  for (i = 0; i < list->count; ++i) {
    if (strcmp(list->items[i], s) == 0) return 1;
  }
  return 0;
}

static void pk_str_list_free(pk_str_list *list) {
  size_t i;
  for (i = 0; i < list->count; ++i) free(list->items[i]);
  free(list->items);
  list->items = NULL;
  list->count = 0;
  list->capacity = 0;
}

static int pk_file_exists(const char *path) {
  struct stat st;
  if (stat(path, &st) != 0) return 0;
  return S_ISREG(st.st_mode) != 0;
}

static char *pk_dirname_owned(const char *path) {
  const char *slash = strrchr(path, PK_PATH_SEP);
  if (!slash) return pk_xstrdup(".");
  if (slash == path) return pk_xstrdup("/");
  {
    size_t n = (size_t)(slash - path);
    char *out = (char *)pk_xmalloc(n + 1);
    memcpy(out, path, n);
    out[n] = '\0';
    return out;
  }
}

static char *pk_join_path(const char *a, const char *b) {
  size_t a_len = strlen(a);
  size_t b_len = strlen(b);
  int need_sep = (a_len > 0 && a[a_len - 1] != PK_PATH_SEP);
  char *out = (char *)pk_xmalloc(a_len + (need_sep ? 1 : 0) + b_len + 1);
  memcpy(out, a, a_len);
  if (need_sep) out[a_len++] = PK_PATH_SEP;
  memcpy(out + a_len, b, b_len);
  out[a_len + b_len] = '\0';
  return out;
}

static char *pk_abspath_or_dup(const char *path) {
#if defined(_WIN32)
  return pk_xstrdup(path);
#else
  if (path[0] == '/') return pk_xstrdup(path);
  {
    char cwd[4096];
    if (!getcwd(cwd, sizeof(cwd))) return pk_xstrdup(path);
    return pk_join_path(cwd, path);
  }
#endif
}

static char *pk_read_line(FILE *f) {
  size_t len = 0;
  size_t cap = 256;
  char *buf = (char *)pk_xmalloc(cap);
  int c;
  while ((c = fgetc(f)) != EOF) {
    if (len + 1 >= cap) {
      cap *= 2;
      buf = (char *)pk_xrealloc(buf, cap);
    }
    buf[len++] = (char)c;
    if (c == '\n') break;
  }
  if (len == 0 && c == EOF) {
    free(buf);
    return NULL;
  }
  buf[len] = '\0';
  return buf;
}

static const char *pk_skip_ws(const char *s) {
  while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n' || *s == '\f' || *s == '\v') ++s;
  return s;
}

static int pk_parse_include(const char *line, char *out_delim, char **out_path) {
  const char *p = pk_skip_ws(line);
  const char *q;
  size_t n;
  if (*p != '#') return 0;
  ++p;
  p = pk_skip_ws(p);
  if (strncmp(p, "include", 7) != 0) return 0;
  p += 7;
  p = pk_skip_ws(p);
  if (*p != '"' && *p != '<') return 0;
  *out_delim = *p;
  ++p;
  q = p;
  while (*q && *q != (*out_delim == '"' ? '"' : '>')) ++q;
  if (!*q) return 0;
  n = (size_t)(q - p);
  *out_path = (char *)pk_xmalloc(n + 1);
  memcpy(*out_path, p, n);
  (*out_path)[n] = '\0';
  return 1;
}

static char *pk_resolve_polykernel_include(const pk_str_list *include_dirs, const char *include_path) {
  size_t i;
  if (strcmp(include_path, "polykernel/dialect.h") == 0) include_path = "polykernel.h";
  for (i = 0; i < include_dirs->count; ++i) {
    char *candidate = pk_join_path(include_dirs->items[i], include_path);
    if (pk_file_exists(candidate)) return candidate;
    free(candidate);
  }
  return NULL;
}

static void pk_render_file(FILE *out,
                           const pk_str_list *include_dirs,
                           pk_str_list *seen_files,
                           const char *path,
                           int emit_line_directives) {
  FILE *f;
  char *canonical = pk_abspath_or_dup(path);
  char *line;
  unsigned long line_no = 0;

  if (pk_str_list_contains(seen_files, canonical)) {
    free(canonical);
    return;
  }
  pk_str_list_push(seen_files, canonical);

  f = fopen(path, "rb");
  if (!f) pk_die_errno(path);

  if (emit_line_directives) fprintf(out, "#line 1 \"%s\"\n", path);

  while ((line = pk_read_line(f)) != NULL) {
    char delim = 0;
    char *inc = NULL;
    ++line_no;

    if (pk_parse_include(line, &delim, &inc)) {
      int is_polykernel =
        (delim == '"' && (strncmp(inc, "polykernel/", 10) == 0 || strcmp(inc, "polykernel.h") == 0));
      if (is_polykernel) {
        char *resolved = pk_resolve_polykernel_include(include_dirs, inc);
        if (!resolved) {
          fprintf(stderr, "render: include not found: \"%s\" (from %s:%lu)\n", inc, path, line_no);
          exit(2);
        }
        pk_render_file(out, include_dirs, seen_files, resolved, emit_line_directives);
        if (emit_line_directives) fprintf(out, "#line %lu \"%s\"\n", line_no + 1, path);
        free(resolved);
        free(inc);
        free(line);
        continue;
      }
      free(inc);
    }

    fputs(line, out);
    free(line);
  }

  if (fclose(f) != 0) pk_die_errno("fclose");
}

static void pk_usage(FILE *out) {
  fprintf(out,
          "usage: render [options] <input>\n"
          "\n"
          "Renders a restricted PolyKernel CUDA-truth kernel source into a single-file\n"
          "OpenCL-friendly source by inlining \"polykernel.h\" and includes under \"polykernel/\".\n"
          "\n"
          "options:\n"
          "  -I <dir>        Add include directory (default: auto-detect repo root, else .)\n"
          "  -o <path>       Write output to file (default: stdout)\n"
          "  --line          Emit #line directives (default)\n"
          "  --no-line       Do not emit #line directives\n"
          "  -h, --help      Show this help\n");
}

static char *pk_find_default_include_dir(const char *input_path) {
  char *input_dir = pk_dirname_owned(input_path);
  char *cursor = pk_abspath_or_dup(input_dir);
  free(input_dir);

  for (;;) {
    {
      char *probe = pk_join_path(cursor, "polykernel.h");
      int ok = pk_file_exists(probe);
      free(probe);
      if (ok) return cursor;
    }
    {
      char *pkg_dir = pk_join_path(cursor, "polykernel");
      char *probe = pk_join_path(pkg_dir, "polykernel.h");
      int ok = pk_file_exists(probe);
      free(probe);
      if (ok) {
        free(cursor);
        return pkg_dir;
      }
      free(pkg_dir);
    }
    {
      char *inc_dir = pk_join_path(cursor, "include");
      char *probe = pk_join_path(inc_dir, "polykernel.h");
      int ok = pk_file_exists(probe);
      free(probe);
      if (ok) {
        free(cursor);
        return inc_dir;
      }
      free(inc_dir);
    }

    if (strcmp(cursor, "/") == 0 || strcmp(cursor, ".") == 0) break;

    {
      char *parent = pk_dirname_owned(cursor);
      if (strcmp(parent, cursor) == 0) {
        free(parent);
        break;
      }
      free(cursor);
      cursor = parent;
    }
  }

  free(cursor);
  return pk_xstrdup(".");
}

int main(int argc, char **argv) {
  pk_str_list include_dirs;
  pk_str_list seen_files;
  const char *input_path = NULL;
  const char *output_path = NULL;
  int emit_line_directives = 1;
  int i;

  memset(&include_dirs, 0, sizeof(include_dirs));
  memset(&seen_files, 0, sizeof(seen_files));

  for (i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
      pk_usage(stdout);
      return 0;
    }
    if (strcmp(arg, "--line") == 0) {
      emit_line_directives = 1;
      continue;
    }
    if (strcmp(arg, "--no-line") == 0) {
      emit_line_directives = 0;
      continue;
    }
    if (strcmp(arg, "-o") == 0) {
      if (i + 1 >= argc) pk_die("missing argument for -o");
      output_path = argv[++i];
      continue;
    }
    if (strcmp(arg, "-I") == 0) {
      if (i + 1 >= argc) pk_die("missing argument for -I");
      pk_str_list_push(&include_dirs, pk_xstrdup(argv[++i]));
      continue;
    }
    if (strncmp(arg, "-I", 2) == 0) {
      pk_str_list_push(&include_dirs, pk_xstrdup(arg + 2));
      continue;
    }
    if (arg[0] == '-') {
      pk_usage(stderr);
      return 2;
    }
    if (input_path) pk_die("multiple input files provided");
    input_path = arg;
  }

  if (!input_path) {
    pk_usage(stderr);
    return 2;
  }

  if (include_dirs.count == 0) {
    pk_str_list_push(&include_dirs, pk_find_default_include_dir(input_path));
  }

  {
    FILE *out = stdout;
    if (output_path) {
      out = fopen(output_path, "wb");
      if (!out) pk_die_errno(output_path);
    }

    pk_render_file(out, &include_dirs, &seen_files, input_path, emit_line_directives);

    if (output_path && fclose(out) != 0) pk_die_errno("fclose");
  }

  pk_str_list_free(&seen_files);
  pk_str_list_free(&include_dirs);
  return 0;
}
