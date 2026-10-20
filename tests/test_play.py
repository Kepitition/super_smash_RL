import configparser

import play


def read(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def test_writes_the_xbox_mapping_for_port_2_and_keeps_other_ports(tmp_path):
    ini = tmp_path / "Config" / "GCPadNew.ini"
    ini.parent.mkdir()
    ini.write_text("[GCPad1]\ndevice = Pipe/0/slippibot1\n")
    assert play.write_xbox_mapping(str(ini), remap=False)
    cfg = read(ini)
    assert cfg.get("GCPad1", "device") == "Pipe/0/slippibot1"
    assert cfg.get("GCPad2", "device") == "XInput/0/Gamepad"
    assert cfg.get("GCPad2", "buttons/a") == "`Button A`"
    assert cfg.get("GCPad2", "main stick/up") == "`Left Y+`"
    assert cfg.get("GCPad2", "triggers/l") == "`Trigger L`"


def test_keeps_a_changed_mapping_unless_asked_to_remap(tmp_path):
    ini = tmp_path / "GCPadNew.ini"
    play.write_xbox_mapping(str(ini), remap=False)
    cfg = read(ini)
    cfg.set("GCPad2", "buttons/a", "`Button B`")  # the player remapped A in Dolphin
    with open(ini, "w") as f:
        cfg.write(f)
    assert not play.write_xbox_mapping(str(ini), remap=False)
    assert read(ini).get("GCPad2", "buttons/a") == "`Button B`"
    assert play.write_xbox_mapping(str(ini), remap=True)
    assert read(ini).get("GCPad2", "buttons/a") == "`Button A`"
