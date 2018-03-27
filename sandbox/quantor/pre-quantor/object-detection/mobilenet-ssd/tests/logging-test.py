import logging


def main():
  logging.basicConfig(level=logging.INFO)
  logging.warning('YMK in logging test warning')
  logging.info('YMK in logging test info')


if __name__ == '__main__':
  main()
