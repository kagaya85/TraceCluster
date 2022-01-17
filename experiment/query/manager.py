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

    def preserve_scenario():
        query_and_preserve(q)

    def payment_scenario():
        query_and_pay(q)

    def cancel_scenario():
        query_and_cancel(q)

    def collect_scenario():
        query_and_collect(q)

    def execute_scenario():
        query_and_execute(q)

    while time.time()-start < timeout:
        query_num = random.randint(2, 10)
        new_login = random_from_weighted({True: 70, False: 30})
        if new_login or q.token == "":
            while not q.login():
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
            q.query_high_speed_ticket_parallel: 10,
            preserve_scenario: 30,
            payment_scenario: 20,
            cancel_scenario: 20,
            collect_scenario: 20,
            execute_scenario: 20,
        }

        for i in range(0, query_num):
            func = random_from_weighted(query_weights)
            logger.info(f'execure query: {func.__name__}')
            try:
                func()
            except Exception:
                logger.exception(f'query {func.__name__} got an exception')

            time.sleep(random.randint(5, 10))

    return


def random_query(q: Query, weights: dict, count: int = random.randint(1, 3), inteval: int = random.randint(10, 20)):
    """
    登陆一个用户并按权重随机发起请求
    :param weights: 权重dict
    :param count: 请求次数
    :param inteval: 请求间隔
    """
    if not q.login():
        return

    for _ in range(0, count):
        func = random_from_weighted(weights)
        logger.info(f'execure query: {func.__name__}')
        try:
            func()
        except Exception:
            logger.exception(f'query {func.__name__} got an exception')

        time.sleep(inteval)

    return


def run(task: Callable, timeout: int):
    start = time.time()
    while time.time() - start < timeout:
        task()
        time.sleep(1)
    return


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
        random_query(q, query_weights)

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
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_route(timeout: int = 1*hour):
    q = Query(url)

    def preserve_scenario():
        query_and_preserve(q)

    query_weights = {
        q.query_high_speed_ticket: 10,
        q.query_min_station: 10,
        q.query_cheapest: 10,
        q.query_quickest: 10,
        preserve_scenario: 50,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_order(timeout: int = 1*hour):
    q = Query(url)

    def payment_scenario():
        query_and_pay(q)

    def cancel_scenario():
        query_and_cancel(q)

    def collect_scenario():
        query_and_collect(q)

    query_weights = {
        q.query_high_speed_ticket: 10,
        q.query_min_station: 10,
        q.query_cheapest: 10,
        q.query_quickest: 10,
        q.query_orders: 20,
        payment_scenario: 50,
        cancel_scenario: 50,
        collect_scenario: 50,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_basic(timeout: int = 1*hour):
    q = Query(url)

    query_weights = {
        q.query_high_speed_ticket: 10,
        q.query_min_station: 10,
        q.query_cheapest: 10,
        q.query_quickest: 10,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_travel_plan(timeout: int = 1*hour):
    q = Query(url)

    query_weights = {
        q.query_min_station: 10,
        q.query_cheapest: 10,
        q.query_quickest: 10,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_station(timeout: int = 1*hour):
    q = Query(url)

    query_weights = {
        q.query_high_speed_ticket: 10,
        q.query_high_speed_ticket_parallel: 10,
        q.query_min_station: 10,
        q.query_cheapest: 10,
        q.query_quickest: 10,
        q.query_orders: 20,
        q.query_other_orders: 20,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_seat(timeout: int = 1*hour):
    q = Query(url)

    def preserve_scenario():
        query_and_preserve(q)

    query_weights = {
        q.query_min_station: 10,
        q.query_cheapest: 10,
        q.query_quickest: 10,
        preserve_scenario: 20,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_config(timeout: int = 1*hour):
    q = Query(url)

    query_weights = {
        q.query_min_station: 10,
        q.query_cheapest: 10,
        q.query_quickest: 10,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_inside_payment(timeout: int = 1*hour):
    q = Query(url)

    def payment_scenario():
        query_and_pay(q)

    query_weights = {
        payment_scenario: 100,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_cancel(timeout: int = 1*hour):
    q = Query(url)

    def cancel_scenario():
        query_and_cancel(q)

    query_weights = {
        cancel_scenario: 100,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_contacts(timeout: int = 1*hour):
    q = Query(url)

    def preserve_scenario():
        query_and_preserve(q)

    query_weights = {
        q.query_contacts: 10,
        preserve_scenario: 10,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_consign(timeout: int = 1*hour):
    q = Query(url)

    def consign_scenario():
        query_and_consign(q)

    query_weights = {
        consign_scenario: 100,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_auth(timeout: int = 1*hour):
    q = Query(url)

    query_weights = {
        q.query_high_speed_ticket: 10,
        q.query_high_speed_ticket_parallel: 10,
        q.query_min_station: 10,
        q.query_cheapest: 10,
        q.query_quickest: 10,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_execute(timeout: int = 1*hour):
    q = Query(url)

    def collect_scenario():
        query_and_collect(q)

    def execute_scenario():
        query_and_execute(q)

    query_weights = {
        collect_scenario: 10,
        execute_scenario: 10,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_preserve(timeout: int = 1*hour):
    q = Query(url)

    def preserve_scenario():
        query_and_preserve(q)

    query_weights = {
        preserve_scenario: 100,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def select_task(idx: int) -> Callable:

    # task for each hour
    tasks = {
        0: query_travel,
        1: query_ticketinfo,
        2: query_route,
        3: query_order,
        4: query_basic,
        5: query_basic,
        6: query_travel_plan,
        7: query_station,
        8: query_seat,
        9: query_config,
        10: query_inside_payment,
        11: query_cancel,
        12: query_contacts,
        13: query_consign,
        14: query_consign,
        15: query_auth,
        16: query_execute,
        17: query_preserve,
        18: query_cancel,
        19: query_cancel,
    }

    if idx not in tasks.keys():
        return None

    return tasks[idx]


def workflow(timeout: int = 24*hour, task_timeout: int = 1*hour):
    start = time.time()
    p = Pool(4)
    last_hour = -1

    logger.info('start constant query')
    p.apply_async(constant_query, args=(timeout,))

    while time.time() - start < timeout:
        current_hour = time.localtime().tm_hour
        task = select_task(current_hour)
        if task == None:
            time.sleep(1*minute)
            continue

        if current_hour != last_hour:
            logger.info(f'execute task: {task.__name__}')
            p.apply_async(task, args=(task_timeout,))
            last_hour = current_hour

        time.sleep(1*minute)

    p.close()
    logger.info('waiting for constant query end...')
    p.join()
    return


def arguments():
    parser = argparse.ArgumentParser(description="query manager arguments")
    parser.add_argument(
        '--duration', help='query constant duration (hour)', default=100)
    parser.add_argument('--url', help='train ticket server url',
                        default='http://175.27.169.178:32677')
    return parser.parse_args()


def main():
    args = arguments()
    global url
    url = args.url
    duration = int(args.duration) * hour
    logger.info(f'start auto-query manager for {duration//hour} hour(s)')

    logger.info('start query workflow')
    workflow(duration)
    logger.info('workflow ended')

    logger.info('auto-query manager ended')


if __name__ == '__main__':
    main()
