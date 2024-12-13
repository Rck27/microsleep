import time


class NonBlockingDelay():
    # """ Non blocking delay class """
    def __init__(self):
        self._timestamp = 0
        self._delay = 0

    def timeout(self):
	# """ Check if time is up """
        return ((millis() - self._timestamp) > self._delay)

    def delay_ms(self, delay):
	# """ Non blocking delay in ms """
        self._timestamp = millis()
        self._delay = delay

def millis():
    """ Get millis """
    return int(time.time() * 1000)

def delay_ms(delay):
    """ Blocking delay in ms """
    t0 = millis()
    while (millis() - t0) < delay:
        pass


t0 = millis()
delay_ms(20)
print(millis() - t0)

d0, d1 = NonBlockingDelay(), NonBlockingDelay()


while True:
    try:
        print("neverending task")
        # if d0.timeout():
        #     print("delay 0")
        #     d0.delay_ms(1000)
        if d1.timeout():
            print("delay 2 ")
            d1.delay_ms(1500)
    except KeyboardInterrupt:
        print("keyboard interupt")
        exit()