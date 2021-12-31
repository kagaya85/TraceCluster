from autoquery import queries


def main():
    q = queries.Query('http://175.27.169.178:32677')
    if q.login():
        ls = q.query_high_speed_ticket()
        if len(ls) > 0:
            print(ls)
        q.query_quickest()

    return


if __name__ == '__main__':
    main()
