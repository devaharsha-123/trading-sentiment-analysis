import argparse
from bot.validators import validate_input
from bot.orders import place_order
from bot.logging_config import setup_logger

setup_logger()

parser = argparse.ArgumentParser(description="Trading Bot CLI")

parser.add_argument("--symbol", required=True)
parser.add_argument("--side", required=True)
parser.add_argument("--type", required=True)
parser.add_argument("--quantity", required=True)
parser.add_argument("--price", required=False)

args = parser.parse_args()

try:
    validate_input(args.symbol, args.side, args.type, args.quantity, args.price)

    print("\n📌 Order Request:")
    print(vars(args))

    order = place_order(
        args.symbol,
        args.side,
        args.type,
        args.quantity,
        args.price
    )

    if order:
        print("\n✅ Order Success:")
        print({
            "orderId": order.get("orderId"),
            "status": order.get("status"),
            "executedQty": order.get("executedQty"),
            "avgPrice": order.get("avgPrice")
        })
    else:
        print("\n❌ Order Failed")

except Exception as e:
    print(f"\n⚠️ Error: {e}")