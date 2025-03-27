"""
Liste des cryptomonnaies à analyser pour le bot de prédiction.
Cette liste contient les principales cryptomonnaies et celles à fort potentiel.
"""

# Liste des principales cryptomonnaies
MAJOR_CRYPTOS = [
    {"symbol": "BTC-USD", "name": "Bitcoin"},
    {"symbol": "ETH-USD", "name": "Ethereum"},
    {"symbol": "BNB-USD", "name": "Binance Coin"},
    {"symbol": "SOL-USD", "name": "Solana"},
    {"symbol": "XRP-USD", "name": "Ripple"},
    {"symbol": "ADA-USD", "name": "Cardano"},
    {"symbol": "AVAX-USD", "name": "Avalanche"},
    {"symbol": "DOGE-USD", "name": "Dogecoin"},
    {"symbol": "DOT-USD", "name": "Polkadot"},
    {"symbol": "MATIC-USD", "name": "Polygon"}
]

# Liste des cryptomonnaies émergentes à fort potentiel
EMERGING_CRYPTOS = [
    {"symbol": "LINK-USD", "name": "Chainlink"},
    {"symbol": "ATOM-USD", "name": "Cosmos"},
    {"symbol": "NEAR-USD", "name": "NEAR Protocol"},
    {"symbol": "FTM-USD", "name": "Fantom"},
    {"symbol": "ALGO-USD", "name": "Algorand"},
    {"symbol": "ICP-USD", "name": "Internet Computer"},
    {"symbol": "FIL-USD", "name": "Filecoin"},
    {"symbol": "HBAR-USD", "name": "Hedera"},
    {"symbol": "VET-USD", "name": "VeChain"},
    {"symbol": "EGLD-USD", "name": "MultiversX"}
]

# Liste des cryptomonnaies de finance décentralisée (DeFi) à fort potentiel
DEFI_CRYPTOS = [
    {"symbol": "UNI-USD", "name": "Uniswap"},
    {"symbol": "AAVE-USD", "name": "Aave"},
    {"symbol": "MKR-USD", "name": "Maker"},
    {"symbol": "CRV-USD", "name": "Curve DAO Token"},
    {"symbol": "COMP-USD", "name": "Compound"},
    {"symbol": "SNX-USD", "name": "Synthetix"},
    {"symbol": "SUSHI-USD", "name": "SushiSwap"},
    {"symbol": "YFI-USD", "name": "yearn.finance"},
    {"symbol": "1INCH-USD", "name": "1inch"},
    {"symbol": "BAL-USD", "name": "Balancer"}
]

# Liste des cryptomonnaies de couche 2 (Layer 2) à fort potentiel
LAYER2_CRYPTOS = [
    {"symbol": "OP-USD", "name": "Optimism"},
    {"symbol": "ARB-USD", "name": "Arbitrum"},
    {"symbol": "IMX-USD", "name": "Immutable X"},
    {"symbol": "LRC-USD", "name": "Loopring"},
    {"symbol": "ZKS-USD", "name": "ZKSpace"}
]

# Toutes les cryptomonnaies à analyser
ALL_CRYPTOS = MAJOR_CRYPTOS + EMERGING_CRYPTOS + DEFI_CRYPTOS + LAYER2_CRYPTOS
