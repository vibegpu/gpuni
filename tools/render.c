#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#endif

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
} gpuni_str_list;

static void gpuni_die(const char *msg) {
  fprintf(stderr, "render: %s\n", msg);
  exit(1);
}

static void gpuni_die_errno(const char *context) {
  fprintf(stderr, "render: %s: %s\n", context, strerror(errno));
  exit(1);
}

static void *gpuni_xmalloc(size_t n) {
  void *p = malloc(n);
  if (!p) gpuni_die("out of memory");
  return p;
}

static void *gpuni_xrealloc(void *p, size_t n) {
  void *q = realloc(p, n);
  if (!q) gpuni_die("out of memory");
  return q;
}

static char *gpuni_xstrdup(const char *s) {
  size_t n = strlen(s) + 1;
  char *p = (char *)gpuni_xmalloc(n);
  memcpy(p, s, n);
  return p;
}

static void gpuni_str_list_push(gpuni_str_list *list, char *owned) {
  if (list->count == list->capacity) {
    size_t new_capacity = list->capacity ? list->capacity * 2 : 8;
    list->items = (char **)gpuni_xrealloc(list->items, new_capacity * sizeof(list->items[0]));
    list->capacity = new_capacity;
  }
  list->items[list->count++] = owned;
}

static int gpuni_str_list_contains(const gpuni_str_list *list, const char *s) {
  size_t i;
  for (i = 0; i < list->count; ++i) {
    if (strcmp(list->items[i], s) == 0) return 1;
  }
  return 0;
}

static void gpuni_str_list_free(gpuni_str_list *list) {
  size_t i;
  for (i = 0; i < list->count; ++i) free(list->items[i]);
  free(list->items);
  list->items = NULL;
  list->count = 0;
  list->capacity = 0;
}

static int gpuni_file_exists(const char *path) {
  struct stat st;
  if (stat(path, &st) != 0) return 0;
  return S_ISREG(st.st_mode) != 0;
}

static char *gpuni_dirname_owned(const char *path) {
  const char *slash = strrchr(path, GU_PATH_SEP);
  if (!slash) return gpuni_xstrdup(".");
  if (slash == path) return gpuni_xstrdup("/");
  {
    size_t n = (size_t)(slash - path);
    char *out = (char *)gpuni_xmalloc(n + 1);
    memcpy(out, path, n);
    out[n] = '\0';
    return out;
  }
}

static char *gpuni_join_path(const char *a, const char *b) {
  size_t a_len = strlen(a);
  size_t b_len = strlen(b);
  int need_sep = (a_len > 0 && a[a_len - 1] != GU_PATH_SEP);
  char *out = (char *)gpuni_xmalloc(a_len + (need_sep ? 1 : 0) + b_len + 1);
  memcpy(out, a, a_len);
  if (need_sep) out[a_len++] = GU_PATH_SEP;
  memcpy(out + a_len, b, b_len);
  out[a_len + b_len] = '\0';
  return out;
}

static char *gpuni_abspath_or_dup(const char *path) {
#if defined(_WIN32)
  return gpuni_xstrdup(path);
#else
  if (path[0] == '/') return gpuni_xstrdup(path);
  {
    char cwd[4096];
    if (!getcwd(cwd, sizeof(cwd))) return gpuni_xstrdup(path);
    return gpuni_join_path(cwd, path);
  }
#endif
}

static char *gpuni_read_line(FILE *f) {
  size_t len = 0;
  size_t cap = 256;
  char *buf = (char *)gpuni_xmalloc(cap);
  int c;
  while ((c = fgetc(f)) != EOF) {
    if (len + 1 >= cap) {
      cap *= 2;
      buf = (char *)gpuni_xrealloc(buf, cap);
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

static const char *gpuni_skip_ws(const char *s) {
  while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n' || *s == '\f' || *s == '\v') ++s;
  return s;
}

static int gpuni_parse_include(const char *line, char *out_delim, char **out_path) {
  const char *p = gpuni_skip_ws(line);
  const char *q;
  size_t n;
  if (*p != '#') return 0;
  ++p;
  p = gpuni_skip_ws(p);
  if (strncmp(p, "include", 7) != 0) return 0;
  p += 7;
  p = gpuni_skip_ws(p);
  if (*p != '"' && *p != '<') return 0;
  *out_delim = *p;
  ++p;
  q = p;
  while (*q && *q != (*out_delim == '"' ? '"' : '>')) ++q;
  if (!*q) return 0;
  n = (size_t)(q - p);
  *out_path = (char *)gpuni_xmalloc(n + 1);
  memcpy(*out_path, p, n);
  (*out_path)[n] = '\0';
  return 1;
}

static char *gpuni_resolve_gpuni_include(const gpuni_str_list *include_dirs, const char *include_path) {
  size_t i;
  if (strcmp(include_path, "gpuni/dialect.h") == 0) include_path = "gpuni.h";
  for (i = 0; i < include_dirs->count; ++i) {
    char *candidate = gpuni_join_path(include_dirs->items[i], include_path);
    if (gpuni_file_exists(candidate)) return candidate;
    free(candidate);
  }
  return NULL;
}

static void gpuni_render_file(FILE *out,
                           const gpuni_str_list *include_dirs,
                           gpuni_str_list *seen_files,
                           const char *path,
                           int emit_line_directives) {
  FILE *f;
  char *canonical = gpuni_abspath_or_dup(path);
  char *line;
  unsigned long line_no = 0;

  if (gpuni_str_list_contains(seen_files, canonical)) {
    free(canonical);
    return;
  }
  gpuni_str_list_push(seen_files, canonical);

  f = fopen(path, "rb");
  if (!f) gpuni_die_errno(path);

  if (emit_line_directives) fprintf(out, "#line 1 \"%s\"\n", path);

  while ((line = gpuni_read_line(f)) != NULL) {
    char delim = 0;
    char *inc = NULL;
    ++line_no;

    if (gpuni_parse_include(line, &delim, &inc)) {
      int is_gpuni = (delim == '"' && (strncmp(inc, "gpuni/", 6) == 0 || strcmp(inc, "gpuni.h") == 0));
      if (is_gpuni) {
        char *resolved = gpuni_resolve_gpuni_include(include_dirs, inc);
        if (!resolved) {
          fprintf(stderr, "render: include not found: \"%s\" (from %s:%lu)\n", inc, path, line_no);
          exit(2);
        }
        gpuni_render_file(out, include_dirs, seen_files, resolved, emit_line_directives);
        if (emit_line_directives) fprintf(out, "#line %lu \"%s\"\n", line_no + 1, path);
        free(resolved);
        free(inc);
        free(line);
        continue;
      }
      free(inc);
    }

    /* Strip 'extern "C" ' prefix (OpenCL C doesn't support it) */
    {
      const char *p = gpuni_skip_ws(line);
      if (strncmp(p, "extern \"C\"", 10) == 0) {
        p += 10;
        p = gpuni_skip_ws(p);
        if (*p == '\0' || *p == '\n' || *p == '\r') {
          /* Line is just 'extern "C"' - skip entirely */
          free(line);
          continue;
        }
        /* Line has content after 'extern "C" ' - output without the prefix */
        fputs(p, out);
        free(line);
        continue;
      }
    }

    fputs(line, out);
    free(line);
  }

  if (fclose(f) != 0) gpuni_die_errno("fclose");
}

/* Extract kernel signatures from rendered source.
   Finds "__global__ [__launch_bounds__(...)] void <name>(...)"
   Returns list of full signatures like "__global__ void saxpy(int n, float* y, ...)" */
static void gpuni_find_kernel_signatures(const char *src, gpuni_str_list *names, gpuni_str_list *sigs) {
  const char *p = src;
  const char *global_kw = "__global__";
  const size_t global_kw_len = 10;
  const char *launch_bounds_kw = "__launch_bounds__";
  const size_t launch_bounds_kw_len = 17;

  /* Match kernel entry points: __global__ [__launch_bounds__] void <name>( */
  while ((p = strstr(p, global_kw)) != NULL) {
    const char *sig_start = p;
    const char *q = p + global_kw_len;
    const char *name_start;
    const char *name_end;
    int paren_depth;
    const char *sig_end;

    q = gpuni_skip_ws(q);
    if (strncmp(q, launch_bounds_kw, launch_bounds_kw_len) == 0) {
      q += launch_bounds_kw_len;
      q = gpuni_skip_ws(q);
      if (*q == '(') {
        paren_depth = 1;
        ++q;
        while (*q && paren_depth > 0) {
          if (*q == '(') paren_depth++;
          else if (*q == ')') paren_depth--;
          ++q;
        }
      }
      q = gpuni_skip_ws(q);
    }

    if (strncmp(q, "void", 4) != 0) {
      p = q;
      continue;
    }
    q += 4;
    q = gpuni_skip_ws(q);

    name_start = q;
    name_end = name_start;

    while (*name_end && ((*name_end >= 'a' && *name_end <= 'z') ||
                         (*name_end >= 'A' && *name_end <= 'Z') ||
                         (*name_end >= '0' && *name_end <= '9') ||
                         *name_end == '_')) ++name_end;
    if (*name_end != '(' || name_end == name_start) {
      p = name_end;
      continue;
    }

    /* Extract name */
    {
      size_t n = (size_t)(name_end - name_start);
      char *name = (char *)gpuni_xmalloc(n + 1);
      memcpy(name, name_start, n);
      name[n] = '\0';
      if (gpuni_str_list_contains(names, name)) {
        free(name);
        p = name_end;
        continue;
      }
      gpuni_str_list_push(names, name);
    }

    /* Find matching closing paren */
    paren_depth = 1;
    sig_end = name_end + 1;
    while (*sig_end && paren_depth > 0) {
      if (*sig_end == '(') paren_depth++;
      else if (*sig_end == ')') paren_depth--;
      sig_end++;
    }

    /* Extract full signature */
    {
      size_t n = (size_t)(sig_end - sig_start);
      char *sig = (char *)gpuni_xmalloc(n + 1);
      memcpy(sig, sig_start, n);
      sig[n] = '\0';
      gpuni_str_list_push(sigs, sig);
    }

    p = sig_end;
  }
}

/* Simplify signature for extern declaration (remove address space qualifiers) */
static char *gpuni_simplify_sig_for_extern(const char *sig) {
  /* Remove GU_GLOBAL/GU_LOCAL/GU_CONSTANT/__global/__local/__constant from signature */
  size_t len = strlen(sig);
  char *out = (char *)gpuni_xmalloc(len + 1);
  const char *p = sig;
  char *q = out;

  while (*p) {
    /* Skip address space qualifiers */
    if (strncmp(p, "GU_GLOBAL ", 10) == 0) { p += 10; continue; }
    if (strncmp(p, "GU_LOCAL ", 9) == 0) { p += 9; continue; }
    if (strncmp(p, "GU_CONSTANT ", 12) == 0) { p += 12; continue; }
    if (strncmp(p, "__global ", 9) == 0) { p += 9; continue; }
    if (strncmp(p, "__local ", 8) == 0) { p += 8; continue; }
    if (strncmp(p, "__constant ", 11) == 0) { p += 11; continue; }
    *q++ = *p++;
  }
  *q = '\0';
  return out;
}

/* Write C header with source string and extern declarations */
static void gpuni_write_header(const char *header_path, const char *src,
                            const gpuni_str_list *kernel_names, const gpuni_str_list *kernel_sigs) {
  FILE *f;
  size_t i;
  char *guard;
  const char *base;
  char *p;

  f = fopen(header_path, "wb");
  if (!f) gpuni_die_errno(header_path);

  /* Generate include guard from filename */
  base = strrchr(header_path, GU_PATH_SEP);
  base = base ? base + 1 : header_path;
  guard = gpuni_xstrdup(base);
  for (p = guard; *p; ++p) {
    if (*p == '.' || *p == '-') *p = '_';
    else if (*p >= 'a' && *p <= 'z') *p = *p - 'a' + 'A';
  }

  fprintf(f, "/* Generated by gpuni/tools/render --emit-header */\n");
  fprintf(f, "#ifndef %s\n", guard);
  fprintf(f, "#define %s\n\n", guard);

  /* CUDA/HIP: extern declarations */
  fprintf(f, "#if !defined(GUH_OPENCL)\n");
  fprintf(f, "#ifdef __cplusplus\n");
  fprintf(f, "extern \"C\" {\n");
  fprintf(f, "#endif\n");
  for (i = 0; i < kernel_sigs->count; ++i) {
    char *simple_sig = gpuni_simplify_sig_for_extern(kernel_sigs->items[i]);
    fprintf(f, "%s;\n", simple_sig);
    free(simple_sig);
  }
  fprintf(f, "#ifdef __cplusplus\n");
  fprintf(f, "}\n");
  fprintf(f, "#endif\n");
  fprintf(f, "#endif\n\n");

  /* Write source string for each kernel (OpenCL only; CUDA/HIP get NULL) */
  for (i = 0; i < kernel_names->count; ++i) {
    const char *name = kernel_names->items[i];
    const char *c;

    fprintf(f, "#if defined(GUH_OPENCL)\n");
    fprintf(f, "static const char %s_gpuni_src[] =\n", name);
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
    fprintf(f, "#define %s_gpuni_src ((const char*)0)\n", name);
    fprintf(f, "#endif\n\n");
  }

  fprintf(f, "#endif /* %s */\n", guard);
  free(guard);

  if (fclose(f) != 0) gpuni_die_errno("fclose");
}

static int gpuni_ends_with(const char *s, const char *suffix) {
  size_t slen = strlen(s);
  size_t sufflen = strlen(suffix);
  if (sufflen > slen) return 0;
  return strcmp(s + slen - sufflen, suffix) == 0;
}

static int gpuni_is_header_path(const char *path) {
  return gpuni_ends_with(path, ".gu.h") || gpuni_ends_with(path, ".h");
}

static void gpuni_usage(FILE *out) {
  fprintf(out,
          "usage: render [options] <input>\n"
          "\n"
          "Renders a gpuni CUDA-truth kernel (*.gu.cu) to a C header or OpenCL source.\n"
          "Output format is determined by file extension:\n"
          "  .gu.h / .h  -> C header with OpenCL source string + CUDA/HIP declarations\n"
          "  .cl         -> raw OpenCL source (for debugging)\n"
          "\n"
          "options:\n"
          "  -I <dir>      Add include directory (default: auto-detect)\n"
          "  -o <path>     Output file (default: stdout as raw OpenCL)\n"
          "  --line        Emit #line directives (default)\n"
          "  --no-line     Do not emit #line directives\n"
          "  -h, --help    Show this help\n"
          "\n"
          "examples:\n"
          "  render kernel.gu.cu -o kernel.gu.h   # C header (recommended)\n"
          "  render kernel.gu.cu -o kernel.cl     # raw OpenCL (for debugging)\n");
}

static char *gpuni_find_default_include_dir(const char *input_path) {
  char *input_dir = gpuni_dirname_owned(input_path);
  char *cursor = gpuni_abspath_or_dup(input_dir);
  free(input_dir);

  for (;;) {
    {
      char *probe = gpuni_join_path(cursor, "gpuni.h");
      int ok = gpuni_file_exists(probe);
      free(probe);
      if (ok) return cursor;
    }
    {
      char *pkg_dir = gpuni_join_path(cursor, "gpuni");
      char *probe = gpuni_join_path(pkg_dir, "gpuni.h");
      int ok = gpuni_file_exists(probe);
      free(probe);
      if (ok) {
        free(cursor);
        return pkg_dir;
      }
      free(pkg_dir);
    }
    {
      char *inc_dir = gpuni_join_path(cursor, "include");
      char *probe = gpuni_join_path(inc_dir, "gpuni.h");
      int ok = gpuni_file_exists(probe);
      free(probe);
      if (ok) {
        free(cursor);
        return inc_dir;
      }
      free(inc_dir);
    }

    if (strcmp(cursor, "/") == 0 || strcmp(cursor, ".") == 0) break;

    {
      char *parent = gpuni_dirname_owned(cursor);
      if (strcmp(parent, cursor) == 0) {
        free(parent);
        break;
      }
      free(cursor);
      cursor = parent;
    }
  }

  free(cursor);
  return gpuni_xstrdup(".");
}

/* Read entire file into buffer */
static char *gpuni_read_file(const char *path) {
  FILE *f = fopen(path, "rb");
  char *buf;
  long len;
  if (!f) gpuni_die_errno(path);
  fseek(f, 0, SEEK_END);
  len = ftell(f);
  fseek(f, 0, SEEK_SET);
  buf = (char *)gpuni_xmalloc((size_t)len + 1);
  if (len > 0 && fread(buf, 1, (size_t)len, f) != (size_t)len) {
    fclose(f);
    gpuni_die_errno(path);
  }
  buf[len] = '\0';
  fclose(f);
  return buf;
}

int main(int argc, char **argv) {
  gpuni_str_list include_dirs;
  gpuni_str_list seen_files;
  const char *input_path = NULL;
  const char *output_path = NULL;
  int emit_line_directives = 1;
  int i;

  memset(&include_dirs, 0, sizeof(include_dirs));
  memset(&seen_files, 0, sizeof(seen_files));

  for (i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
      gpuni_usage(stdout);
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
      if (i + 1 >= argc) gpuni_die("missing argument for -o");
      output_path = argv[++i];
      continue;
    }
    if (strcmp(arg, "-I") == 0) {
      if (i + 1 >= argc) gpuni_die("missing argument for -I");
      gpuni_str_list_push(&include_dirs, gpuni_xstrdup(argv[++i]));
      continue;
    }
    if (strncmp(arg, "-I", 2) == 0) {
      gpuni_str_list_push(&include_dirs, gpuni_xstrdup(arg + 2));
      continue;
    }
    if (arg[0] == '-') {
      gpuni_usage(stderr);
      return 2;
    }
    if (input_path) gpuni_die("multiple input files provided");
    input_path = arg;
  }

  if (!input_path) {
    gpuni_usage(stderr);
    return 2;
  }

  if (include_dirs.count == 0) {
    gpuni_str_list_push(&include_dirs, gpuni_find_default_include_dir(input_path));
  }

  /* Determine output mode: header (.h/.gu.h) or raw OpenCL (.cl/stdout) */
  {
    int output_header = 0;
    const char *final_header_path = NULL;

    /* Detect header output from -o extension */
    if (output_path && gpuni_is_header_path(output_path)) {
      output_header = 1;
      final_header_path = output_path;
    }

    if (output_header) {
      /* Render to temp, then generate header */
      char *temp_path = gpuni_xstrdup("/tmp/gpuni_render_XXXXXX");
      FILE *out;
      char *src;
      gpuni_str_list kernel_names;
      gpuni_str_list kernel_sigs;

      {
        int fd = mkstemp(temp_path);
        if (fd < 0) gpuni_die_errno("mkstemp");
        out = fdopen(fd, "wb");
        if (!out) { close(fd); gpuni_die_errno("fdopen"); }
      }

      gpuni_render_file(out, &include_dirs, &seen_files, input_path, emit_line_directives);
      if (fclose(out) != 0) gpuni_die_errno("fclose");

      src = gpuni_read_file(temp_path);
      memset(&kernel_names, 0, sizeof(kernel_names));
      memset(&kernel_sigs, 0, sizeof(kernel_sigs));
      gpuni_find_kernel_signatures(src, &kernel_names, &kernel_sigs);
      if (kernel_names.count == 0) {
        fprintf(stderr, "render: warning: no kernels found (__global__ void <name>(...))\n");
      }
      gpuni_write_header(final_header_path, src, &kernel_names, &kernel_sigs);
      gpuni_str_list_free(&kernel_names);
      gpuni_str_list_free(&kernel_sigs);
      free(src);
      unlink(temp_path);
      free(temp_path);
    } else {
      /* Raw OpenCL output */
      FILE *out = stdout;
      if (output_path) {
        out = fopen(output_path, "wb");
        if (!out) gpuni_die_errno(output_path);
      }
      gpuni_render_file(out, &include_dirs, &seen_files, input_path, emit_line_directives);
      if (output_path && fclose(out) != 0) gpuni_die_errno("fclose");
    }
  }

  gpuni_str_list_free(&seen_files);
  gpuni_str_list_free(&include_dirs);
  return 0;
}
