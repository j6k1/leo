use std::cmp;
use usiagent::rule::{LegalMove, Rule, Square, State};
use usiagent::shogi::KomaKind::{Blank, SFu, SGin, SKaku, SHisha, SKin, GFu, GGin, GKin, GKaku, GHisha};
use usiagent::shogi::{KomaKind, ObtainKind, Teban};
use crate::solver::Fraction;

#[inline]
pub fn initial_pn_dn_plus_or_node(teban:Teban,state:&State,m:LegalMove) -> (Fraction,Fraction) {
    let mut pn = 1;
    let mut dn = 1;

    let to = match m {
        LegalMove::To(m) => m.dst() as Square,
        LegalMove::Put(m) => m.dst() as Square
    };

    let attack_support = Rule::control_count(teban,state,to);
    let defense_support = Rule::control_count(teban.opposite(),state,to);

    if defense_support >= 2 {
        pn += 1;
    }

    let bonus = match m {
        LegalMove::Put(_) => 1,
        _ => 0
    };

    let x = to / 9;
    let y = to - x * 9;

    let obtained = state.get_banmen().0[y as usize][x as usize];

    if attack_support + bonus > defense_support {
        dn += 1;
    } else if obtained != Blank {
        if obtained == SKin || obtained == GKin || obtained == SGin || obtained == GGin {
            dn += 1;
        } else {
            pn += 1;
        }
    } else {
        pn += 1;
    }

    (Fraction::new(pn),Fraction::new(dn))
}
#[inline]
pub fn initial_pn_dn_plus_and_node(teban:Teban,state:&State,m:LegalMove) -> (Fraction,Fraction) {
    match m {
        LegalMove::To(m) if m.obtained().is_some() => {
            return (Fraction::new(2), Fraction::new(1))
        },
        _ => ()
    };

    let to = match m {
        LegalMove::To(m) => m.dst() as Square,
        LegalMove::Put(m) => m.dst() as Square
    };

    match m {
        LegalMove::To(m) if m.src() as Square == Rule::ou_square(teban, state) => {
            return (Fraction::new(1), Fraction::new(1));
        },
        _ => ()
    };

    let attack_support = Rule::control_count(teban, state, to);
    let defense_support = Rule::control_count(teban.opposite(), state, to);

    let bonus = match m {
        LegalMove::Put(_) => 1,
        _ => 0
    };

    if attack_support < defense_support + bonus {
        return (Fraction::new(2),Fraction::new(1));
    }

    (Fraction::new(1),Fraction::new(2))
}
#[inline]
pub fn calc_asc_priority(teban:Teban,state:&State,m:LegalMove) -> i32 {
    const KPT_VALUES:[i32;29] = [
        1, 2, 2, 3, 5, 5, 5, 8, 5, 5, 5, 5, 8, 8,
        1, 2, 2, 3, 5, 5, 5, 8, 5, 5, 5, 5, 8, 8,
        0
    ];

    let mut value = 0;

    match m {
        LegalMove::To(m) if !m.is_nari() => {
            let x = m.src() / 9;
            let y = m.src() - x * 9;

            let kind = state.get_banmen().0[y as usize][x as usize];

            match kind {
                SFu | SKaku | SHisha |
                GFu | GKaku | GHisha if Rule::is_possible_nari(kind,m.src() as Square,m.dst() as Square) => {
                    value += 100;
                },
                _ => ()
            }
        },
        _ => ()
    }

    let kind = match m {
        LegalMove::To(m) if m.is_nari() => {
            let x = m.src() / 9;
            let y = m.src() - x * 9;

            state.get_banmen().0[y as usize][x as usize].to_nari() as usize
        },
        LegalMove::To(m) => {
            let x = m.src() / 9;
            let y = m.src() - x * 9;

            state.get_banmen().0[y as usize][x as usize] as usize
        },
        LegalMove::Put(m) => {
            m.kind() as usize
        }
    };

    value -= KPT_VALUES[kind];

    match m {
        LegalMove::To(m) => {
            value += dist(Rule::ou_square(teban,state),m.dst() as Square);
        },
        LegalMove::Put(m) => {
            value += dist(Rule::ou_square(teban,state),m.dst() as Square);
        }
    }

    value
}
#[inline]
pub fn attack_priority(teban:Teban,state:&State,m:LegalMove) -> i32 {
    const KPT_VALUES:[i32;29] = [
        1, 2, 2, 3, 5, 5, 5, 8, 5, 5, 5, 5, 8, 8,
        1, 2, 2, 3, 5, 5, 5, 8, 5, 5, 5, 5, 8, 8,
        0
    ];

    if Rule::is_oute_move(state,teban,m) {
        let mut value = 0;

        match m {
            LegalMove::To(m) if !m.is_nari() => {
                let x = m.src() / 9;
                let y = m.src() - x * 9;

                let kind = state.get_banmen().0[y as usize][x as usize];

                match kind {
                    SFu | SKaku | SHisha |
                    GFu | GKaku | GHisha if Rule::is_possible_nari(kind, m.src() as Square, m.dst() as Square) => {
                        value += 100;
                    },
                    _ => ()
                }
            },
            _ => ()
        }

        let kind = match m {
            LegalMove::To(m) if m.is_nari() => {
                let x = m.src() / 9;
                let y = m.src() - x * 9;

                state.get_banmen().0[y as usize][x as usize].to_nari() as usize
            },
            LegalMove::To(m) => {
                let x = m.src() / 9;
                let y = m.src() - x * 9;

                state.get_banmen().0[y as usize][x as usize] as usize
            },
            LegalMove::Put(m) => {
                m.kind() as usize
            }
        };

        value -= KPT_VALUES[kind];

        match m {
            LegalMove::To(m) => {
                value += dist(Rule::ou_square(teban, state), m.dst() as Square);
            },
            LegalMove::Put(m) => {
                value += dist(Rule::ou_square(teban, state), m.dst() as Square);
            }
        }

        value
    } else {
        let mut value = 0;

        let to = match m {
            LegalMove::To(m) => m.dst() as Square,
            LegalMove::Put(m) => m.dst() as Square
        };

        let attack_support = Rule::control_count(teban,state,to);
        let defense_support = Rule::control_count(teban.opposite(),state,to);

        if defense_support >= 2 {
            value += 1;
        }

        let bonus = match m {
            LegalMove::Put(_) => 1,
            _ => 0
        };

        let x = to / 9;
        let y = to - x * 9;

        let obtained = state.get_banmen().0[y as usize][x as usize];

        if attack_support + bonus > defense_support {
            value -= 1;
        } else if obtained != Blank {
            if obtained == SKin || obtained == GKin || obtained == SGin || obtained == GGin {
                value -= 1;
            } else {
                value += 1;
            }
        } else {
            value += 1;
        }

        value
    }
}
#[inline]
pub fn defense_priority(_:Teban,state:&State,m:LegalMove) -> i32 {
    match m {
        LegalMove::Put(m) => {
            m.kind() as i32 + KomaKind::SHishaN as i32 + ObtainKind::HishaN as i32 + 4
        },
        LegalMove::To(m) if m.obtained() == Some(ObtainKind::Ou) => {
            0
        },
        LegalMove::To(m)=> {
            let src = m.src();
            let x = src / 9;
            let y = src - x * 9;
            let kind = state.get_banmen().0[y as usize][x as usize];

            match m.obtained() {
                Some(o) => {
                    o as i32 - ObtainKind::HishaN as i32 + 1
                },
                None if kind == KomaKind::SOu || kind == KomaKind::GOu => {
                    ObtainKind::HishaN as i32 + 2
                },
                None => {
                    ObtainKind::HishaN as i32 + 3
                }
            }
        }
    }
}
#[inline]
fn dist(o:Square,to:Square) -> i32 {
    let ox = o / 9;
    let oy = o - ox * 9;
    let tx = to / 9;
    let ty = to - tx * 9;

    cmp::max((ox - tx).abs(),(oy - ty).abs())
}