import logging

def place_order(symbol, side, order_type, quantity, price=None):
    order = {
        "orderId": 123456,
        "status": "FILLED",
        "executedQty": quantity,
        "avgPrice": price if price else "market_price"
    }

    logging.info(f"Order placed: {order}")   # ✅ THIS LINE IS KEY

    print("Simulated Order Executed")
    return order