from pylsl import resolve_streams
ss = resolve_streams(wait_time=3)
if not ss:
    print("No streams found.")
else:
    for i in ss:
        print(f"Name='{i.name()}', Type='{i.type()}', Ch={i.channel_count()}, Fs={i.nominal_srate()}")
        print(i)
