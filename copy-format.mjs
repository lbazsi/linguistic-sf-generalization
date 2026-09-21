// Postgres COPY ... FROM STDIN text format (the default, tab-separated) helpers.
// NULL is represented by the literal two characters \N. Backslash, tab, newline and
// carriage return must be backslash-escaped inside a field.

export function escapeCopyValue(value) {
  if (value === null || value === undefined) return "\\N";
  const str =
    value instanceof Date
      ? value.toISOString()
      : typeof value === "object"
      ? JSON.stringify(value)
      : String(value);
  return str.replace(/\\/g, "\\\\").replace(/\t/g, "\\t").replace(/\n/g, "\\n").replace(/\r/g, "\\r");
}

export function formatCopyRow(values) {
  return values.map(escapeCopyValue).join("\t") + "\n";
}
