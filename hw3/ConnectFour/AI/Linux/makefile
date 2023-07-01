.PHONY: compete strategy compete_debug strategy_debug clean
all: compete strategy

compete:
	$(MAKE) -C Compete

strategy:
	$(MAKE) -C Strategy so

debug: compete_debug strategy_debug

compete_debug:
	$(MAKE) -C Compete debug

strategy_debug:
	$(MAKE) -C Strategy debug

clean:
	$(MAKE) -C Strategy clean
	$(MAKE) -C Compete clean
