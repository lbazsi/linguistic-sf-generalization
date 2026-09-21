-- Example schema for trying the pipeline locally. Load this into an empty Postgres database
-- and point SOURCE_DATABASE_URL at it (TARGET_DATABASE_URL should point at a second, empty
-- database with the SAME schema loaded but no data).

CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  email VARCHAR(255) NOT NULL UNIQUE,
  full_name VARCHAR(255) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  manager_id INTEGER REFERENCES users(id) -- self-referential FK, exercises stage 2/6
);

CREATE TABLE products (
  id SERIAL PRIMARY KEY,
  sku VARCHAR(64) NOT NULL UNIQUE,
  name VARCHAR(255) NOT NULL,
  price NUMERIC(10, 2) NOT NULL,
  in_stock BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE orders (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id),
  status VARCHAR(32) NOT NULL DEFAULT 'pending',
  placed_at TIMESTAMP NOT NULL DEFAULT now(),
  shipping_notes TEXT
);

CREATE TABLE order_items (
  id SERIAL PRIMARY KEY,
  order_id INTEGER NOT NULL REFERENCES orders(id),
  product_id INTEGER NOT NULL REFERENCES products(id),
  quantity INTEGER NOT NULL CHECK (quantity > 0),
  unit_price NUMERIC(10, 2) NOT NULL
);
