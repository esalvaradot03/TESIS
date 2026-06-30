schtasks /create /tn "TesisLiveTrading" /tr "C:\dev\Tesis\scripts\run_live_trading.bat" /sc weekly /d MON,TUE,WED,THU,FRI /st 11:00 /f
