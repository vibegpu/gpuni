#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#if defined(_WIN32)
#  include <direct.h>
#  define GU_PATH_SEP '\\'
#else
#  include <unistd.h>
#  define GU_PATH_SEP '/'
#endif

typedef struct {
  char **items;
  size_t count;
  size_t capacity;
} gu_str_list;

static void gu_die(const char *msg) {
  fprintf(stderr, "render: %s\n", msg);
  exit(1);
}

static void gu_die_errno(const char *context) {
  fprintf(stderr, "render: %s: %s\n", context, strerror(errno));
  exit(1);
}

static void *gu_xmalloc(size_t n) {
  void *p = malloc(n);
  if (!p) gu_die("out of memory");
  return p;
}

static void *gu_xrealloc(void *p, size_t n) {
  void *q = realloc(p, n);
  if (!q) gu_die("out of memory");
  return q;
}

static char *gu_xstrdup(const char *s) {
  size_t n = strlen(s) + 1;
  char *p = (char *)gu_xmalloc(n);
  memcpy(p, s, n);
  return p;
}

static void gu_str_list_push(gu_str_list *list, char *owned) {
  if (list->count == list->capacity) {
    size_t new_capacity = list->capacity ? list->capacity * 2 : 8;
    list->items = (char **)gu_xrealloc(list->items, new_capacity * sizeof(list->items[0]));
    list->capacity = new_capacity;
  }
  list->items[list->count++] = owned;
}

static int gu_str_list_contains(const gu_str_list *list, const char *s) {
  size_t i;
  for (i = 0; i < list->count; ++i) {
    if (strcmp(list->items[i], s) == 0) return 1;
  }
  return 0;
}

static void gu_str_list_free(gu_str_list *list) {
  size_t i;
  for (i = 0; i < list->count; ++i) free(list->items[i]);
  free(list->items);
  list->items = NULL;
  list->count = 0;
  list->capacity = 0;
}

static int gu_file_exists(const char *path) {
  struct stat st;
  if (stat(path, &st) != 0) return 0;
  return S_ISREG(st.st_mode) != 0;
}

static char *gu_dirname_owned(const char *path) {
  const char *slash = strrchr(path, GU_PATH_SEP);
  if (!slash) return gu_xstrdup(".");
  if (slash == path) return gu_xstrdup("/");
  {
    size_t n = (size_t)(slash - path);
    char *out = (char *)gu_xmalloc(n + 1);
    memcpy(out, path, n);
    out[n] = '\0';
    return out;
  }
}

static char *gu_join_path(const char *a, const char *b) {
  size_t a_len = strlen(a);
  size_t b_len = strlen(b);
  int need_sep = (a_len > 0 && a[a_len - 1] != GU_PATH_SEP);
  char *out = (char *)gu_xmalloc(a_len + (need_sep ? 1 : 0) + b_len + 1);
  memcpy(out, a, a_len);
  if (need_sep) out[a_len++] = GU_PATH_SEP;
  memcpy(out + a_len, b, b_len);
  out[a_len + b_len] = '\0';
  return out;
}

static char *gu_abspath_or_dup(const char *path) {
#if defined(_WIN32)
  return gu_xstrdup(path);
#else
  if (path[0] == '/') return gu_xstrdup(path);
  {
    char cwd[4096];
    if (!getcwd(cwd, sizeof(cwd))) return gu_xstrdup(path);
    return gu_join_path(cwd, path);
  }
#endif
}

static char *gu_read_line(FILE *f) {
  size_t len = 0;
  size_t cap = 256;
  char *buf = (char *)gu_xmalloc(cap);
  int c;
  while ((c = fgetc(f)) != EOF) {
    if (len + 1 >= cap) {
      cap *= 2;
      buf = (char *)gu_xrealloc(buf, cap);
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

static const char *gu_skip_ws(const char *s) {
  while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n' || *s == '\f' || *s == '\v') ++s;
  return s;
}

static int gu_parse_include(const char *line, char *out_delim, char **out_path) {
  const char *p = gu_skip_ws(line);
  const char *q;
  size_t n;
  if (*p != '#') return 0;
  ++p;
  p = gu_skip_ws(p);
  if (strncmp(p, "include", 7) != 0) return 0;
  p += 7;
  p = gu_skip_ws(p);
  if (*p != '"' && *p != '<') return 0;
  *out_delim = *p;
  ++p;
  q = p;
  while (*q && *q != (*out_delim == '"' ? '"' : '>')) ++q;
  if (!*q) return 0;
  n = (size_t)(q - p);
  *out_path = (char *)gu_xmalloc(n + 1);
  memcpy(*out_path, p, n);
  (*out_path)[n] = '\0';
  return 1;
}

static char *gu_resolve_gpuni_include(const gu_str_list *include_dirs, const char *include_path) {
  size_t i;
  if (strcmp(include_path, "gpuni/dialect.h") == 0) include_path = "gpuni.h";
  for (i = 0; i < include_dirs->count; ++i) {
    char *candidate = gu_join_path(include_dirs->items[i], include_path);
    if (gu_file_exists(candidate)) return candidate;
    free(candidate);
  }
  return NULL;
}

static void gu_render_file(FILE *out,
                           const gu_str_list *include_dirs,
                           gu_str_list *seen_files,
                           const char *path,
                           int emit_line_directives) {
  FILE *f;
  char *canonical = gu_abspath_or_dup(path);
  char *line;
  unsigned long line_no = 0;

  if (gu_str_list_contains(seen_files, canonical)) {
    free(canonical);
    return;
  }
  gu_str_list_push(seen_files, canonical);

  f = fopen(path, "rb");
  if (!f) gu_die_errno(path);

  if (emit_line_directives) fprintf(out, "#line 1 \"%s\"\n", path);

  while ((line = gu_read_line(f)) != NULL) {
    char delim = 0;
    char *inc = NULL;
    ++line_no;

    if (gu_parse_include(line, &delim, &inc)) {
      int is_gpuni = (delim == '"' && (strncmp(inc, "gpuni/", 6) == 0 || strcmp(inc, "gpuni.h") == 0));
      if (is_gpuni) {
        char *resolved = gu_resolve_gpuni_include(include_dirs, inc);
        if (!resolved) {
          fprintf(stderr, "render: include not found: \"%s\" (from %s:%lu)\n", inc, path, line_no);
          exit(2);
        }
        gu_render_file(out, include_dirs, seen_files, resolved, emit_line_directives);
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

  if (fclose(f) != 0) gu_die_errno("fclose");
}

/* Extract kernel names from rendered source (finds "__global__ void gu_<name>(") */
static void gu_find_kernel_names(const char *src, gu_str_list *names) {
  const char *p = src;
  /* Match kernel entry points: __global__ void gu_<name>( */
  while ((p = strstr(p, "__global__ void gu_")) != NULL) {
    const char *start = p + 16;  /* skip "__global__ void " */
    const char *end = start;
    while (*end && ((*end >= 'a' && *end <= 'z') ||
                    (*end >= 'A' && *end <= 'Z') ||
                    (*end >= '0' && *end <= '9') ||
                    *end == '_')) ++end;
    if (*end == '(' && end > start) {
      size_t n = (size_t)(end - start);
      char *name = (char *)gu_xmalloc(n + 1);
      memcpy(name, start, n);
      name[n] = '\0';
      if (!gu_str_list_contains(names, name)) {
        gu_str_list_push(names, name);
      } else {
        free(name);
      }
    }
    p = end;
  }
}

/* Write C header with source string */
static void gu_write_header(const char *header_path, const char *src, const gu_str_list *kernel_names) {
  FILE *f;
  size_t i;
  char *guard;
  const char *base;
  char *p;

  f = fopen(header_path, "wb");
  if (!f) gu_die_errno(header_path);

  /* Generate include guard from filename */
  base = strrchr(header_path, GU_PATH_SEP);
  base = base ? base + 1 : header_path;
  guard = gu_xstrdup(base);
  for (p = guard; *p; ++p) {
    if (*p == '.' || *p == '-') *p = '_';
    else if (*p >= 'a' && *p <= 'z') *p = *p - 'a' + 'A';
  }

  fprintf(f, "/* Generated by gpuni/tools/render --emit-header */\n");
  fprintf(f, "#ifndef %s\n", guard);
  fprintf(f, "#define %s\n\n", guard);

  /* Write source string for each kernel (OpenCL only; CUDA/HIP get NULL) */
  for (i = 0; i < kernel_names->count; ++i) {
    const char *name = kernel_names->items[i];
    const char *c;

    fprintf(f, "#if defined(GUH_OPENCL)\n");
    fprintf(f, "static const char %s_gu_source[] =\n", name);
    fprintf(f, "  \"");
    for (c = src; *c; ++c) {
      switch (*c) {
        case '\\': fputs("\\\\", f); break;
        case '"':  fputs("\\\"", f); break;
        case '\n': fputs("\\n\"\n  \"", f); break;
        case '\r': break;  /* skip CR */
        case '\t': fputs("\\t", f); break;
        default:   fputc(*c, f); break;
      }
    }
    fprintf(f, "\";\n");
    fprintf(f, "#else\n");
    fprintf(f, "/* CUDA/HIP: source not needed, GU_KERNEL ignores this */\n");
    fprintf(f, "#define %s_gu_source ((const char*)0)\n", name);
    fprintf(f, "#endif\n\n");
  }

  fprintf(f, "#endif /* %s */\n", guard);
  free(guard);

  if (fclose(f) != 0) gu_die_errno("fclose");
}

static void gu_usage(FILE *out) {
  fprintf(out,
          "usage: render [options] <input>\n"
          "\n"
          "Renders a restricted gpuni CUDA-truth kernel source into a single-file\n"
          "OpenCL-friendly source by inlining \"gpuni.h\" and includes under \"gpuni/\".\n"
          "\n"
          "options:\n"
          "  -I <dir>        Add include directory (default: auto-detect repo root, else .)\n"
          "  -o <path>       Write output to file (default: stdout)\n"
          "  --emit-header <path>  Also emit C header with source string\n"
          "  --line          Emit #line directives (default)\n"
          "  --no-line       Do not emit #line directives\n"
          "  -h, --help      Show this help\n");
}

static char *gu_find_default_include_dir(const char *input_path) {
  char *input_dir = gu_dirname_owned(input_path);
  char *cursor = gu_abspath_or_dup(input_dir);
  free(input_dir);

  for (;;) {
    {
      char *probe = gu_join_path(cursor, "gpuni.h");
      int ok = gu_file_exists(probe);
      free(probe);
      if (ok) return cursor;
    }
    {
      char *pkg_dir = gu_join_path(cursor, "gpuni");
      char *probe = gu_join_path(pkg_dir, "gpuni.h");
      int ok = gu_file_exists(probe);
      free(probe);
      if (ok) {
        free(cursor);
        return pkg_dir;
      }
      free(pkg_dir);
    }
    {
      char *inc_dir = gu_join_path(cursor, "include");
      char *probe = gu_join_path(inc_dir, "gpuni.h");
      int ok = gu_file_exists(probe);
      free(probe);
      if (ok) {
        free(cursor);
        return inc_dir;
      }
      free(inc_dir);
    }

    if (strcmp(cursor, "/") == 0 || strcmp(cursor, ".") == 0) break;

    {
      char *parent = gu_dirname_owned(cursor);
      if (strcmp(parent, cursor) == 0) {
        free(parent);
        break;
      }
      free(cursor);
      cursor = parent;
    }
  }

  free(cursor);
  return gu_xstrdup(".");
}

/* Read entire file into buffer */
static char *gu_read_file(const char *path) {
  FILE *f = fopen(path, "rb");
  char *buf;
  long len;
  if (!f) gu_die_errno(path);
  fseek(f, 0, SEEK_END);
  len = ftell(f);
  fseek(f, 0, SEEK_SET);
  buf = (char *)gu_xmalloc((size_t)len + 1);
  if (len > 0 && fread(buf, 1, (size_t)len, f) != (size_t)len) {
    fclose(f);
    gu_die_errno(path);
  }
  buf[len] = '\0';
  fclose(f);
  return buf;
}

int main(int argc, char **argv) {
  gu_str_list include_dirs;
  gu_str_list seen_files;
  const char *input_path = NULL;
  const char *output_path = NULL;
  const char *header_path = NULL;
  int emit_line_directives = 1;
  int i;

  memset(&include_dirs, 0, sizeof(include_dirs));
  memset(&seen_files, 0, sizeof(seen_files));

  for (i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
      gu_usage(stdout);
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
      if (i + 1 >= argc) gu_die("missing argument for -o");
      output_path = argv[++i];
      continue;
    }
    if (strcmp(arg, "--emit-header") == 0) {
      if (i + 1 >= argc) gu_die("missing argument for --emit-header");
      header_path = argv[++i];
      continue;
    }
    if (strcmp(arg, "-I") == 0) {
      if (i + 1 >= argc) gu_die("missing argument for -I");
      gu_str_list_push(&include_dirs, gu_xstrdup(argv[++i]));
      continue;
    }
    if (strncmp(arg, "-I", 2) == 0) {
      gu_str_list_push(&include_dirs, gu_xstrdup(arg + 2));
      continue;
    }
    if (arg[0] == '-') {
      gu_usage(stderr);
      return 2;
    }
    if (input_path) gu_die("multiple input files provided");
    input_path = arg;
  }

  if (!input_path) {
    gu_usage(stderr);
    return 2;
  }

  if (header_path && !output_path) {
    gu_die("--emit-header requires -o <output>");
  }

  if (include_dirs.count == 0) {
    gu_str_list_push(&include_dirs, gu_find_default_include_dir(input_path));
  }

  /* Render to .cl file */
  {
    FILE *out = stdout;
    if (output_path) {
      out = fopen(output_path, "wb");
      if (!out) gu_die_errno(output_path);
    }

    gu_render_file(out, &include_dirs, &seen_files, input_path, emit_line_directives);

    if (output_path && fclose(out) != 0) gu_die_errno("fclose");
  }

  /* Generate header if requested */
  if (header_path) {
    char *src = gu_read_file(output_path);
    gu_str_list kernel_names;
    memset(&kernel_names, 0, sizeof(kernel_names));
    gu_find_kernel_names(src, &kernel_names);
    if (kernel_names.count == 0) {
      fprintf(stderr, "render: warning: no kernels found (void gu_*)\n");
    }
    gu_write_header(header_path, src, &kernel_names);
    gu_str_list_free(&kernel_names);
    free(src);
  }

  gu_str_list_free(&seen_files);
  gu_str_list_free(&include_dirs);
  return 0;
}
