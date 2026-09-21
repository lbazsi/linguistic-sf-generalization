import { faker } from "@faker-js/faker";

function resolveFakerMethod(methodPath) {
  const parts = methodPath.split(".");
  let fn = faker;
  for (const part of parts) {
    fn = fn?.[part];
  }
  if (typeof fn !== "function") {
    throw new Error(`Unknown faker method path: "${methodPath}"`);
  }
  return fn.bind(parts.length > 1 ? faker[parts[0]] : faker);
}

function weightedPick(values, weights) {
  if (!weights || !weights.length) {
    return values[Math.floor(Math.random() * values.length)];
  }
  const total = weights.reduce((a, b) => a + b, 0);
  let r = Math.random() * total;
  for (let i = 0; i < values.length; i++) {
    r -= weights[i];
    if (r <= 0) return values[i];
  }
  return values[values.length - 1];
}

function randomDateBetween(from, to) {
  const fromMs = new Date(from).getTime();
  const toMs = new Date(to).getTime();
  return new Date(fromMs + Math.random() * (toMs - fromMs));
}

/** Fallback used when the AI spec has no entry for a column (e.g. spec predates a new column). */
export function defaultStrategyForColumn(column) {
  const type = column.dataType.toLowerCase();
  if (type.includes("bool")) return { type: "boolean", trueRate: 0.5 };
  if (type.includes("uuid")) return { type: "uuid" };
  if (type.includes("timestamp")) return { type: "timestampRecent", withinDays: 365 };
  if (type === "date") return { type: "date", from: "2020-01-01", to: "2025-12-31" };
  if (type.includes("int") || type.includes("numeric") || type.includes("decimal") || type.includes("real") || type.includes("double")) {
    return { type: "number", min: 0, max: 1000, decimals: type.includes("int") ? 0 : 2 };
  }
  if (type.includes("json")) return { type: "constant", value: {} };
  return { type: "faker", method: "lorem.words", args: [3] };
}

export function valueForStrategy(strategy, column) {
  switch (strategy.type) {
    case "faker": {
      const fn = resolveFakerMethod(strategy.method);
      let value = fn(...(strategy.args || []));
      if (column.maxLength && typeof value === "string" && value.length > column.maxLength) {
        value = value.slice(0, column.maxLength);
      }
      return value;
    }
    case "enum":
      return weightedPick(strategy.values, strategy.weights);
    case "boolean":
      return Math.random() < (strategy.trueRate ?? 0.5);
    case "number": {
      const raw = strategy.min + Math.random() * (strategy.max - strategy.min);
      const decimals = strategy.decimals ?? 0;
      return Number(raw.toFixed(decimals));
    }
    case "date":
      return column.dataType.toLowerCase() === "date"
        ? randomDateBetween(strategy.from, strategy.to).toISOString().slice(0, 10)
        : randomDateBetween(strategy.from, strategy.to).toISOString();
    case "timestampRecent":
      return faker.date.recent({ days: strategy.withinDays ?? 30 }).toISOString();
    case "constant":
      return strategy.value;
    case "uuid":
      return faker.string.uuid();
    default:
      throw new Error(`Unknown strategy type: "${strategy.type}"`);
  }
}

/** Generates a value, retrying on collision for columns that must be unique. */
export function valueForColumn(column, strategy, seenValues) {
  if (!column.isUnique) return valueForStrategy(strategy, column);

  const maxAttempts = 25;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const value = valueForStrategy(strategy, column);
    const key = String(value);
    if (!seenValues.has(key)) {
      seenValues.add(key);
      return value;
    }
  }
  // Exhausted retries (small enum-backed "unique" columns, etc.) — disambiguate deterministically
  // rather than violating the unique constraint on load.
  const fallback = `${valueForStrategy(strategy, column)}-${seenValues.size}`;
  seenValues.add(fallback);
  return fallback;
}
