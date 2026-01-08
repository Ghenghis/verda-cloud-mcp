# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| 1.x.x   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public issue
2. Email the maintainers directly or use GitHub's private vulnerability reporting
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes

## Security Best Practices

### API Credentials

- **Never** commit `config.yaml` with real credentials
- Use environment variables for production
- Rotate credentials regularly
- Use minimal required permissions

```bash
# Use environment variables
export VERDA_CLIENT_ID="your-id"
export VERDA_CLIENT_SECRET="your-secret"
```

### SSH Keys

- Use strong SSH keys (Ed25519 or RSA 4096-bit)
- Never share private keys
- Use key passphrases for additional security
- Rotate keys periodically

### Instance Security

- Keep instances updated
- Use startup scripts to configure firewalls
- Limit SSH access to known IPs when possible
- Monitor running instances regularly

## Known Security Considerations

### Spot Instance Data

Spot instances can be evicted at any time. Ensure:
- Checkpoints are saved frequently
- Sensitive data is not left on instances
- Volume data is encrypted if needed

### Webhook Security

When using webhooks for notifications:
- Use HTTPS endpoints only
- Validate webhook signatures if available
- Don't include sensitive data in webhook payloads

## Security Updates

Security updates are released as patch versions (e.g., 2.0.1).
Subscribe to releases to stay informed.
