# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x | :white_check_mark: |
| < 0.1 | :x: |

## Reporting a Vulnerability

If you believe you have found a security vulnerability, please do **not** open a public issue. Instead, report it privately by:

1. Emailing security concerns to `marcus@example.com`
2. Describing the vulnerability with as much detail as possible
3. Including steps to reproduce if applicable

We will respond within 48 hours and keep you updated on the remediation process.

## Security Best Practices

When using this library:

- Never commit `.env` files with credentials
- Use environment variables for all secrets
- Rotate Redis and database credentials regularly
- Restrict network access to Redis and PostgreSQL instances
- Use TLS/SSL for production deployments

## Dependencies

This project relies on external dependencies. Monitor these for security advisories:

- dspy-ai
- langgraph
- redis
- psycopg

Keep dependencies updated via `uv sync`.
