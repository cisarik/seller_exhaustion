# Strategy Export System Guide

**Version**: 2.1  
**Last Updated**: 2025-01-15  
**Purpose**: Export optimized strategies from backtesting tool to live trading agent

---

## 🎯 Overview

This guide explains how to use the Strategy Export System to deploy your backtested strategies to a live trading agent.

### What is Strategy Export?

The **Strategy Export System** bridges the gap between:
1. **Backtesting Tool** (this application) - Research, optimize, validate strategies
2. **Trading Agent** (separate application) - Execute strategies on live markets

**Key Concept**: You develop and optimize in the backtesting tool, then export a single JSON file containing ALL necessary information for the trading agent to execute your strategy.

---

## 📦 What Gets Exported?

A complete strategy configuration file (`config.json`) containing:

### 1. Strategy Parameters (Entry Logic)
- EMA periods (fast/slow)
- Z-score windows and thresholds
- ATR window and multipliers
- Close location minimum
- **All time-based parameters auto-converted for timeframe**

### 2. Exit Configuration
- Fibonacci exits (enabled/disabled, lookback, lookahead, target level)
- Stop-loss (enabled/disabled, ATR multiplier)
- Traditional TP (enabled/disabled, R:R ratio)
- Time exit (enabled/disabled, max hold bars)

### 3. Risk Management (Safe Defaults)
- Risk per trade: 1% of account
- Max position size: 10% of account
- Max daily loss: 5%
- Max daily trades: 10
- Max open positions: 1
- Slippage tolerance: 0.5%
- Order timeout: 30 seconds

### 4. Exchange Configuration (Placeholders)
- Exchange name: binance (default)
- Trading pair: ADA/USDT (default)
- **API credentials: PLACEHOLDER (configure in agent)**
- Testnet: ENABLED (default)
- Paper trading: ENABLED (default)
- Initial balance: $10,000 (paper trading)

### 5. Data Feed Configuration
- Data source: exchange (default)
- WebSocket: enabled (default)
- REST API fallback settings
- Data validation settings

### 6. Backtest Metrics (Reference)
- Total trades
- Win rate
- Average R-multiple
- Total PnL
- Max drawdown
- Sharpe ratio

---

## 🚀 Quick Start

### Step 1: Optimize Your Strategy

1. Open the backtesting tool
2. Download historical data (Settings → Data Download)
3. Configure strategy parameters (sidebar or Strategy Editor)
4. Run backtest to validate
5. **Optional**: Use genetic algorithm to optimize parameters

### Step 2: Export Strategy

1. Click **💾 Export Strategy** in toolbar
2. Choose save location (default: `strategy_[timeframe]_[dates].json`)
3. Review validation warnings:
   - ✅ **Green**: No issues
   - ⚠️ **Orange**: Warnings (review but OK to proceed)
   - ❌ **Red**: Critical issues (fix before deploying)

**Common Warnings** (expected on export):
- ⚠️ API key not configured (normal - configure in agent)
- ⚠️ API secret not configured (normal - configure in agent)

### Step 3: Deploy to Trading Agent

**See PRD_TRADING_AGENT.md for complete setup instructions.**

Quick overview:
1. Copy exported `strategy_*.json` to trading agent directory
2. Rename to `config.json`
3. Create `.env` file with real exchange credentials
4. Start trading agent in **paper trading mode**
5. Monitor for 7+ days before considering live trading

---

## 📝 Export Examples

### Example 1: Default Strategy Export

```bash
# After backtesting with default parameters
Click: Export Strategy
File: strategy_15m_2024-01-01_2024-12-31.json

# Result: 
✅ Exported with paper trading ENABLED
✅ Exported with testnet ENABLED
⚠️ API credentials need configuration
```

### Example 2: Optimized Strategy Export

```bash
# After genetic algorithm optimization
1. Run 50 generations of GA
2. Click "Apply Best Parameters"
3. Re-run backtest to validate
4. Click: Export Strategy
File: strategy_15m_optimized_gen50.json

# Result:
✅ Exported with backtest metrics (50 trades, 67% WR, 0.8 avg R)
⚠️ High risk per trade: 2.0% (review recommended)
```

### Example 3: Conservative Strategy Export

```bash
# Using Conservative fitness preset
1. Set fitness preset to "🛡️ Conservative"
2. Optimize with GA (min 50% win rate)
3. Verify results: 15 trades, 73% WR, low DD
4. Export Strategy
File: strategy_15m_conservative.json

# Result:
✅ Low risk configuration
✅ High win rate optimization
✅ Ready for paper trading
```

---

## 🔍 Validation Checks

When you export, the system validates your configuration:

### ✅ Safe Configuration
```
No warnings
- Paper trading enabled
- Testnet enabled  
- Risk per trade ≤ 2%
- Max daily loss ≤ 10%
```

### ⚠️ Configuration Warnings
```
⚠️ API key not configured (using placeholder)
⚠️ API secret not configured (using placeholder)
ℹ️ 1-minute timeframe - Very active trading

Action: Review warnings, acceptable for export
```

### ❌ Critical Issues
```
⚠️ LIVE TRADING ENABLED - Real money at risk!
⚠️ Production exchange + live trading - EXTREME RISK!
⚠️ High risk per trade: 5%
⚠️ High max daily loss: 25%

Action: Fix configuration before deploying!
```

---

## 📥 Importing Strategies

### Why Import?

- **Load saved strategies** from previous sessions
- **Share strategies** with other users
- **Test different configs** without re-optimizing

### How to Import

1. Click **📥 Import Strategy** in toolbar
2. Select `*.json` file
3. Review validation warnings
4. Click Yes to load parameters into UI
5. **Run backtest** to verify on your data

**Important**: Importing only loads parameters into UI, does NOT start trading.

### Import Validation

```
✅ Strategy Imported Successfully
Strategy: Seller Exhaustion
Timeframe: 15m
Description: Optimized 15m strategy with 61.8% Fib exits

⚠️ 2 Warnings:
  - API credentials are placeholders
  - Paper trading enabled

✓ Parameters loaded into UI
→ Click 'Run Backtest' to test on your data
```

---

## ⚙️ Configuration Details

### File Format: JSON

**Location**: Anywhere (you choose on export)  
**Suggested naming**: `strategy_[timeframe]_[description].json`  
**Size**: ~2-5 KB (small, text-based)

**Example**:
```json
{
  "version": "2.1.0",
  "created_at": "2025-01-15T10:30:00Z",
  "description": "Optimized 15m strategy with Fibonacci exits",
  "strategy_name": "Seller Exhaustion",
  "timeframe": "15m",
  
  "seller_params": {
    "ema_fast": 96,
    "ema_slow": 672,
    "vol_z": 2.0,
    ...
  },
  
  "backtest_params": {
    "use_fib_exits": true,
    "fib_target_level": 0.618,
    ...
  },
  
  "risk_management": {
    "risk_per_trade_percent": 1.0,
    "max_daily_loss_percent": 5.0,
    ...
  },
  
  "exchange": {
    "exchange_name": "binance",
    "trading_pair": "ADA/USDT",
    "api_key": "YOUR_API_KEY_HERE",
    "paper_trading": true,
    ...
  }
}
```

### Editing Exported Files

**Safe to edit**:
- `description` - Add notes
- `risk_management.*` - Adjust risk limits
- `exchange.trading_pair` - Change pair (ADA/USDT → ADA/USD)
- `exchange.exchange_name` - Change exchange

**⚠️ Do NOT edit** (unless you know what you're doing):
- `seller_params.*` - Changes strategy logic
- `backtest_params.*` - Changes exit behavior
- `version` - Compatibility issues

**🚫 NEVER edit directly**:
- `api_key`, `api_secret` - Configure in agent's `.env` file instead

---

## 🔒 Security Best Practices

### ❌ What NOT to Do

```bash
# BAD: Real credentials in config.json
{
  "exchange": {
    "api_key": "aBcDeFgHiJkLmN...real_key",  # ❌ NEVER!
    "api_secret": "zYxWvU...real_secret"     # ❌ NEVER!
  }
}

# BAD: Committing config with credentials
git add config.json  # ❌ If contains real keys
git commit -m "Add strategy"
git push  # ❌ Keys now public!
```

### ✅ What TO Do

```bash
# GOOD: Placeholders in exported config.json
{
  "exchange": {
    "api_key": "YOUR_API_KEY_HERE",        # ✅ Placeholder
    "api_secret": "YOUR_API_SECRET_HERE"   # ✅ Placeholder
  }
}

# GOOD: Real credentials in agent's .env file (never committed)
# In trading agent directory:
# .env file:
EXCHANGE_API_KEY=aBcDeFgHiJkLmN...real_key
EXCHANGE_API_SECRET=zYxWvU...real_secret

# .gitignore file:
.env              # ✅ Never committed
config.json       # ✅ Safe to commit (no real keys)
```

### Security Checklist

- [ ] Exported config has placeholder API keys
- [ ] Real API keys only in agent's `.env` file
- [ ] `.env` in `.gitignore`
- [ ] API keys have IP whitelist enabled (exchange settings)
- [ ] API keys have withdrawal disabled
- [ ] Paper trading enabled for testing
- [ ] Testnet enabled for initial deployment

---

## 🧪 Testing Before Live

### Phase 1: Paper Trading (7+ days)

```bash
# config.json settings:
{
  "exchange": {
    "paper_trading": true,        # ✅ No real orders
    "testnet": true,              # ✅ Use testnet data
    "paper_initial_balance": 10000.0
  }
}

# Monitor:
- Trade execution accuracy
- Signal detection timing
- Slippage simulation
- Connection stability
```

### Phase 2: Testnet Trading (7+ days)

```bash
# config.json settings:
{
  "exchange": {
    "paper_trading": false,       # Real orders (testnet)
    "testnet": true,              # ✅ Still testnet!
    "api_key": "testnet_key_here"
  }
}

# Monitor:
- Real order execution
- Exchange API integration
- Fee calculations
- Error handling
```

### Phase 3: Live Trading (When Ready)

```bash
# config.json settings:
{
  "exchange": {
    "paper_trading": false,       # ⚠️ Real orders
    "testnet": false,             # ⚠️ Production!
    "api_key": "production_key_here"
  }
}

# ONLY proceed if:
✅ 14+ days combined paper + testnet testing
✅ Win rate matches backtest (±10%)
✅ No critical errors
✅ Comfortable with risk
```

---

## 🔄 Update Workflow

### When to Re-Export

**Re-export when**:
- Strategy parameters changed
- Risk limits adjusted  
- Exit configuration modified
- Switching timeframes
- Performance degrades (need re-optimization)

**No need to re-export when**:
- Exchange credentials change (update agent's `.env` only)
- Different trading pair (edit `config.json` directly)
- Risk % tweaks (edit `config.json` directly)

### Versioning Strategies

```bash
# Good naming convention:
strategy_15m_v1.0_baseline.json
strategy_15m_v1.1_opt_gen50.json
strategy_15m_v2.0_fib_618.json
strategy_15m_v2.1_conservative.json

# Track what worked:
- v1.0: 45 trades, 56% WR (baseline)
- v1.1: 67 trades, 62% WR (optimized) ← Best so far
- v2.0: 23 trades, 74% WR (conservative)
```

---

## ❓ FAQ

### Q: Can I edit the exported JSON file manually?

**A**: Yes, but be careful:
- ✅ Safe: description, risk_management.*, exchange.trading_pair
- ⚠️ Risky: seller_params.*, backtest_params.* (changes strategy)
- 🚫 Never: API credentials (use agent's `.env` instead)

### Q: Do I need to re-export after every backtest?

**A**: No. Only export when you're happy with the parameters and ready to deploy or save the configuration.

### Q: Can I share my exported strategy?

**A**: Yes! The exported file contains NO sensitive information (API keys are placeholders). Safe to share via git, email, etc.

### Q: What if I import a strategy and the backtest results are different?

**A**: Normal. Different data ranges or market conditions produce different results. Always backtest on YOUR data before deploying.

### Q: Can I export multiple strategies for different timeframes?

**A**: Yes! Export one config per timeframe:
```
strategy_1m_scalping.json
strategy_15m_balanced.json  
strategy_1h_conservative.json
```
Each runs in a separate trading agent instance.

### Q: What happens if I export with live trading enabled?

**A**: The system warns you loudly:
```
⚠️ LIVE TRADING ENABLED - Real money at risk!
⚠️ Production exchange + live trading - EXTREME RISK!
```
**Recommendation**: Always export with `paper_trading: true` and `testnet: true`. Enable live trading only in the agent after thorough testing.

---

## 📚 Additional Resources

- **PRD_TRADING_AGENT.md**: Complete specifications for the trading agent application
- **AGENTS.md**: Development guide for the backtesting tool
- **STRATEGY_DEFAULTS_GUIDE.md**: Understanding default parameters
- **README.md**: Feature overview and quick start

---

## ✅ Export Checklist

Before deploying an exported strategy, verify:

**Backtesting Phase**:
- [ ] Downloaded sufficient historical data (7+ days)
- [ ] Ran backtest with satisfactory results
- [ ] Validated strategy logic (entry/exit conditions)
- [ ] Reviewed backtest metrics (win rate, drawdown, trade count)
- [ ] Optional: Optimized with genetic algorithm

**Export Phase**:
- [ ] Exported strategy to JSON
- [ ] Reviewed validation warnings (all green or acceptable orange)
- [ ] Saved file with descriptive name
- [ ] Verified paper trading = true
- [ ] Verified testnet = true

**Deployment Phase** (in trading agent):
- [ ] Copied config.json to agent directory
- [ ] Created .env with real (testnet) credentials
- [ ] Started agent in paper trading mode
- [ ] Monitored for 7+ days
- [ ] Verified performance matches backtest (±10%)
- [ ] No critical errors or connection issues

**Live Trading Phase** (only after extensive testing):
- [ ] 14+ days combined testing
- [ ] Win rate stable and matches expectations
- [ ] Comfortable with risk management settings
- [ ] Emergency stop procedures documented
- [ ] Monitoring/alerts configured
- [ ] Small capital allocation initially

---

**Remember**: 
- **Backtest → Optimize → Export → Paper Test → Testnet → Live (gradually)**
- **Start small, scale slowly**
- **Monitor constantly, adjust as needed**
- **When in doubt, stay in paper trading**

---

**END OF STRATEGY EXPORT GUIDE**

**Version**: 2.1  
**Status**: ✅ Complete  
**Last Updated**: 2025-01-15
