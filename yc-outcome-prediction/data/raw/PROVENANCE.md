# Data Provenance

- **Dataset**: Y Combinator companies (open-source mirror of the YC directory)
- **Source repo**: https://github.com/yc-oss/api
- **Download URLs**:
  - https://raw.githubusercontent.com/yc-oss/api/main/companies/all.json -> `yc_companies_all.json`
  - https://raw.githubusercontent.com/yc-oss/api/main/meta.json -> `yc_meta.json`
- **Download date**: 2026-07-02 (originally acquired 2026-07-02)
- **Records**: 5999 companies
- **Notes**: The canonical API host (yc-oss.github.io) was blocked by the network
  proxy, so files were fetched from the repo's `main` branch via
  raw.githubusercontent.com. The repo auto-updates daily from YC's Algolia index;
  `yc_meta.json.last_updated` records the upstream refresh timestamp.
- **License/terms**: Data originates from Y Combinator's public directory
  (https://www.ycombinator.com/companies); yc-oss/api is an unofficial mirror.
