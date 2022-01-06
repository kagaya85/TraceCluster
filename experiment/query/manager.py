from typing import Callable
from autoquery.queries import Query
from autoquery.scenarios import *
from autoquery.utils import random_from_weighted
import logging
import random
import time
import argparse
from multiprocessing import Process, Pool

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger("autoquery-manager")

url = 'http://175.27.169.178:32677'
minute = 60
hour = 60*minute


def constant_query(timeout: int = 24*hour):
    start = time.time()
    q = Query(url)

    while time.time()-start < timeout:
        query_num = random.randint(2, 20)
        new_login = random_from_weighted({True: 70, False: 30})
        if new_login or q.token == "":
            if not q.login():
                logger.error('login failed')
                time.sleep(10)
                continue

        query_weights = {
            q.query_cheapest: 20,
            q.query_orders: 30,
            q.query_food: 5,
            q.query_high_speed_ticket: 50,
            q.query_contacts: 10,
            q.query_min_station: 20,
            q.query_quickest: 20,
            # q.query_route: 10,
            q.query_high_speed_ticket_parallel: 10,
        }

        for i in range(0, query_num):
            func = random_from_weighted(query_weights)
            logger.info(f'execure query: {func.__name__}')
            try:
                func()
            except Exception:
                logger.exception(f'query {func.__name__} got an exception')

            time.sleep(random.randint(5, 30))

    return


def random_query(q: Query, weights: dict, count: int):
    """
    登陆一个用户并按权重随机发起请求
    """
    if not q.login():
        logger.error('login failed')
        return

    for _ in range(0, count):
        func = random_from_weighted(weights)
        logger.info(f'execure query: {func.__name__}')
        try:
            func()
        except Exception:
            logger.exception(f'query {func.__name__} got an exception')

        time.sleep(random.randint(1, 10))


def run(task: Callable, timeout: int):
    start = time.time()
    while time.time() - start < timeout:
        task()
        time.sleep(1)
    return


def select_task(idx: int) -> Callable:
    def query_travel(timeout: int = 1*hour):
        q = Query(url)

        def preserve_scenario():
            query_and_preserve(q)

        query_weights = {
            q.query_high_speed_ticket: 10,
            q.query_high_speed_ticket_parallel: 10,
            q.query_min_station: 10,
            q.query_cheapest: 10,
            q.query_quickest: 10,
            preserve_scenario: 50,
        }

        def task():
            random_query(q, query_weights, random.randint(2, 10))

        run(task, timeout)
        return

    def query_ticketinfo(timeout: int = 1*hour):
        q = Query(url)

        def preserve_scenario():
            query_and_preserve(q)

        query_weights = {
            q.query_high_speed_ticket: 10,
            q.query_high_speed_ticket_parallel: 10,
            q.query_min_station: 10,
            q.query_cheapest: 10,
            q.query_quickest: 10,
            preserve_scenario: 50,
        }

        def task():
            random_query(q, query_weights, random.randint(2, 10))

        run(task, timeout)
        return

    def query_route(timeout: int = 1*hour):

        return

    tasks = {
        0: query_travel,
        1: query_ticketinfo,
    }

    if idx not in tasks.keys():
        return None

    return tasks[idx]


def workflow(timeout: int = 24*hour, task_timeout: int = 1*hour):
    start = time.time()
    p = Pool(4)
    last_hour = -1

    while time.time() - start < timeout:
        current_hour = time.localtime().tm_hour
        task = select_task(current_hour)
        if task == None:
            time.sleep(1*minute)
            continue

        if current_hour != last_hour:
            logger.info(f'execute task {task.__name__}')
            p.apply_async(task, args=(task_timeout))
            last_hour = current_hour

        time.sleep(1*minute)

    p.close()
    p.join()
    return


def arguments():
    parser = argparse.ArgumentParser(description="query manager arguments")
    parser.add_argument(
        '--duration', help='query constant duration (hour)', default=24)
    return parser.parse_args()


def main():
    args = arguments()
    duration = args.duration * hour
    logger.info('start auto-query manager')
    p = Process(traget=constant_query, args=(duration))
    logger.info('start constant query')
    p.start()

    logger.info('start query workflow')
    workflow(duration)
    logger.info('workflow ended')

    logger.info('waiting for constant query end...')
    p.join()
    logger.info('auto-query manager ended')


if __name__ == '__main__':
    main()
