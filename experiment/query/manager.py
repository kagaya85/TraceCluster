from autoquery.queries import Query
from autoquery.scenarios import *
from autoquery.utils import random_from_weighted
import logging
import random
import time

url = 'http://175.27.169.178:32677'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autoquery-manager")


def test():
    q = Query(url)
    if not q.login():
        logger.fatal('login failed')

    ls = q.query_high_speed_ticket()
    if len(ls) > 0:
        print(ls)
    q.query_quickest()

    return


def constant_query():
    q = Query(url)

    while True:
        query_num = random.randint(2, 20)
        new_login = random_from_weighted({True: 70, False: 30})
        if new_login or q.token == "":
            if not q.login():
                time.sleep(10)
                continue

        query_weights = {
            q.query_cheapest: 20,
            q.query_orders: 10,
            q.query_food: 5,
            q.query_high_speed_ticket: 50,
            q.query_contacts: 10,
            q.query_min_station: 20,
            q.query_quickest: 20,
            q.query_route: 10,
            q.query_high_speed_ticket_parallel: 10,
        }

        for i in range(0, query_num):
            func = random_from_weighted(query_weights)
            logger.info(f'execure query {func.__name__}')
            func()
            time.sleep(random.randint(5, 30))


def main():
    logger.info('start auto-query manager')
    constant_query()
    logger.info('auto-query end')


if __name__ == '__main__':
    main()
