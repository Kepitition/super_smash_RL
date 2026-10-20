from smash_rl.env import SmashEnv


def test_instant_restart_only_for_one_stage_and_a_fixed_cpu(tmp_path):
    assert SmashEnv().dolphin.instant_restart
    assert not SmashEnv(stages=("FINAL_DESTINATION", "BATTLEFIELD")).dolphin.instant_restart
    assert not SmashEnv(opponent="RANDOM").dolphin.instant_restart


def test_saving_replays_goes_through_the_menus(tmp_path):
    # Slippi only finishes writing a replay when the game ends normally; an instantly
    # restarted game leaves it "in progress" and unreadable.
    assert not SmashEnv(save_replays=str(tmp_path)).dolphin.instant_restart


def test_parallel_games_save_replays_in_their_own_folders(tmp_path):
    # Slippi names replays by the second: games ending together would overwrite each other.
    a = SmashEnv(port=51441, save_replays=str(tmp_path)).dolphin.save_replays
    b = SmashEnv(port=51442, save_replays=str(tmp_path)).dolphin.save_replays
    assert a != b and a.startswith(str(tmp_path)) and b.startswith(str(tmp_path))
