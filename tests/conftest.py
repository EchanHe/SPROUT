def pytest_addoption(parser):
    parser.addoption("--full", action="store_true", help="run the slow checks without their --quick shortcut")
