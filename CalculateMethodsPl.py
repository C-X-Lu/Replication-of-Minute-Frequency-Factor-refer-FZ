import pandas as pd
import polars as pl
import statsmodels.api as sm
from tqdm import tqdm
from joblib import Parallel, delayed
import torch

"""
    =========================
        方正金工分钟频因子   
    =========================
"""


# 复用函数

def r_mean_20_ts_pl(col_name: str) -> pl.Expr:
    """时序滚动20日均值"""
    return (
        pl.col(col_name)
        .rolling_mean(20, min_samples=20)
        .over('code')
        .alias(f'{col_name}_mean')
    )


def r_std_20_ts_pl(col_name: str) -> pl.Expr:
    """时序滚动20日标准差"""
    return (
        pl.col(col_name)
        .rolling_std(20, min_samples=20, ddof=0)
        .over('code')
        .alias(f'{col_name}_std')
    )


def standardize_cs_pl(col_name: str) -> pl.Expr:
    """截面标准化"""
    return (
        pl.when(
            pl.col(col_name).std(ddof=0) != 0
        ).then(
            (
                    pl.col(col_name) - pl.col(col_name).mean()
            ) / pl.col(col_name).std(ddof=0)
        ).otherwise(
            pl.col(col_name) - pl.col(col_name).mean()
        ).over('date')
        .alias(col_name)
    )


# 适度冒险：耀眼波动率和耀眼收益率的20日均值与20日标准差4个因子等权合成

def cal_YaoYanBoDongLv(df: pl.DataFrame):
    """
    适度冒险因子的中间因子：耀眼波动率
    :param df:
    :return:
    """
    return (
        df.lazy().with_columns(
            pl.col('volume')
            .diff()
            .over(['code', 'date'])
            .alias('vol_diff'),
            (pl.col('close') / pl.col('open') - 1)
            .alias('pct_change')
        ).with_columns(
            (
                pl.col('pct_change')
                .shift(-i)
                .over(['code', 'date'])
                .alias(f'pct_change_shift_-{i}')
            ) for i in range(0, 5)
        ).filter(
            (
                    pl.col('time') > 93500000
            ) & (
                    pl.col('time') < 145300000
            )
        ).with_columns(
            pl.when(
                pl.col('vol_diff') > (
                        pl.col('vol_diff').mean()
                        + pl.col('vol_diff').std(ddof=0)
                ).over(['code', 'date'])
            )
            .then(
                pl.concat_arr(
                    pl.col(f'pct_change_shift_-{i}') for i in range(0, 5)
                ).arr.std(ddof=0)
            )
            .otherwise(None)
            .alias('YaoYanBoDongLv')
        ).group_by(['code', 'date']).agg(
            pl.col('YaoYanBoDongLv')
            .mean()
            .fill_null(0)
        ).with_columns(
            (
                    pl.col('YaoYanBoDongLv')
                    - pl.col('YaoYanBoDongLv').mean()
            )
            .over('date')
            .alias('YaoYanBoDongLv')
        ).collect()
    )


def cal_YaoYanShouYiLv(df: pl.DataFrame):
    """
    适度冒险因子的中间因子：耀眼收益率
    :param df:
    :return:
    """
    return (
        df.lazy().with_columns(
            pl.col('volume')
            .diff()
            .over(['code', 'date'])
            .alias('vol_diff'),
            (pl.col('close') / pl.col('open') - 1)
            .alias('pct_change')
        ).filter(
            (
                    pl.col('time') > 93500000
            ) & (
                    pl.col('time') < 145300000
            )
        ).with_columns(
            pl.when(
                pl.col('vol_diff') > (
                        pl.col('vol_diff').mean()
                        + pl.col('vol_diff').std(ddof=0)
                ).over(['code', 'date'])
            )
            .then(1)
            .otherwise(0)
            .alias('raise_moment')
        ).with_columns(
            pl.when(pl.col('raise_moment') == 1)
            .then(
                pl.col('pct_change')
            )
            .otherwise(None)
            .alias('YaoYanShouYiLv')
        ).group_by(['code', 'date']).agg(
            pl.col('YaoYanShouYiLv')
            .mean()
            .fill_null(0)
        ).with_columns(
            (
                    pl.col('YaoYanShouYiLv')
                    - pl.col('YaoYanShouYiLv').mean()
            ).over('date')
            .alias('YaoYanShouYiLv')
        ).collect()
    )


def cal_ShiDuMaoXian(df: pl.DataFrame):
    """
    适度冒险计算方法
    :param df:
    :return:
    """
    return (
        df.lazy().sort(by=['code', 'date'])
        .with_columns(
            (
                    r_mean_20_ts_pl('YaoYanBoDongLv') + r_std_20_ts_pl('YaoYanBoDongLv')
            ).alias('YaoYanBoDongLv'),
            (
                    r_mean_20_ts_pl('YaoYanShouYiLv') + r_std_20_ts_pl('YaoYanShouYiLv')
            ).alias('YaoYanShouYiLv')
        ).with_columns(
            standardize_cs_pl('YaoYanBoDongLv'),
            standardize_cs_pl('YaoYanShouYiLv')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (pl.col('YaoYanBoDongLv') + pl.col('YaoYanShouYiLv'))
            .alias('ShiDuMaoXian')
        ).collect()
    )


# 潮汐因子：强势半潮汐的20日均值和弱势半潮汐的20日标准差等权合成

def cal_QiangShiBanChaoXi(df: pl.DataFrame):
    """
    潮汐因子的中间因子：强势半潮汐
    :param df:
    :return:
    """
    time_expr = (
            pl.col('time') // 10000000 * 60
            + pl.col('time') % 10000000 / 100000
    ).cast(pl.Int64)
    trade_minute_expr = (
        pl.when(time_expr < 720)
        .then(time_expr - 570)
        .otherwise(time_expr - 660)
    )
    return (
        df.lazy().filter(
            pl.col('close') != 0
        ).with_columns(
            trade_minute_expr
            .cast(pl.Int64)
            .alias('minute_in_trade')
        ).filter(
            (
                    pl.col('minute_in_trade') > 5
            ) & (
                    pl.col('minute_in_trade') < 233
            )
        ).with_columns(
            pl.col('volume')
            .rolling_sum(window_size=9, center=True)
            .over(['code', 'date'])
            .alias('volume_beside')
        ).with_columns(
            pl.col('close')
            .get(
                pl.col('volume_beside')
                .arg_max()
            )
            .over(['code', 'date'])
            .alias('peak_close'),
            pl.col('minute_in_trade')
            .get(
                pl.col('volume_beside')
                .arg_max()
            )
            .over(['code', 'date'])
            .alias('peak_moment')
        ).with_columns(
            pl.when(
                pl.col('minute_in_trade') < pl.col('peak_moment')  # 涨潮段
            )
            .then(
                (
                        pl.col('peak_close') / pl.col('close') - 1
                ) / (
                        pl.col('peak_moment') - pl.col('minute_in_trade')
                )
            )
            .otherwise(
                (
                        pl.col('close') / pl.col('peak_close') - 1
                ) / (
                        pl.col('minute_in_trade') - pl.col('peak_moment')
                )
            )
            .alias('rate')
        ).group_by(['code', 'date']).agg(
            pl.col('rate')
            .get(
                pl.col('volume_beside')
                .arg_min()
            )
            .alias('QiangShiBanChaoXi')
        ).collect()
    )


def cal_RuoShiBanChaoXi(df: pl.DataFrame):
    """
    潮汐因子的中间因子：弱势半潮汐
    :param df:
    :return:
    """
    time_expr = (
            pl.col('time') // 10000000 * 60
            + pl.col('time') % 10000000 / 100000
    ).cast(pl.Int64)
    trade_minute_expr = (
        pl.when(time_expr < 720)
        .then(time_expr - 570)
        .otherwise(time_expr - 660)
    )
    return (
        df.filter(
            pl.col('close') != 0
        ).with_columns(
            trade_minute_expr
            .cast(pl.Int64)
            .alias('minute_in_trade')
        ).filter(
            (
                    pl.col('minute_in_trade') > 5
            ) & (
                    pl.col('minute_in_trade') < 233
            )
        ).with_columns(
            pl.col('volume')
            .rolling_sum(window_size=9, center=True)
            .over(['code', 'date'])
            .alias('volume_beside')
        ).with_columns(
            pl.when(
                pl.col('volume_beside') > (
                    pl.col('volume_beside')
                    .max()
                    .over(['code', 'date'])
                )
            )
            .then(1)
            .otherwise(0)
            .alias('peak'),  # 顶峰时刻
            pl.col('close')
            .get(
                pl.col('volume_beside')
                .arg_max()
            )
            .over(['code', 'date'])
            .alias('peak_close'),
            pl.col('minute_in_trade')
            .get(
                pl.col('volume_beside')
                .arg_max()
            )
            .over(['code', 'date'])
            .alias('peak_moment')
        ).with_columns(
            pl.col('peak')
            .sort_by(by='minute_in_trade')
            .cum_sum()
            .over(['code', 'date'])
            .alias('ebb')  # 退潮
        ).with_columns(
            pl.when(
                pl.col('minute_in_trade') < pl.col('peak_moment')  # 涨潮段
            )
            .then(
                (
                        pl.col('peak_close') / pl.col('close') - 1
                ) / (
                        pl.col('peak_moment') - pl.col('minute_in_trade')
                )
            )
            .otherwise(
                (
                        pl.col('close') / pl.col('peak_close') - 1
                ) / (
                        pl.col('minute_in_trade') - pl.col('peak_moment')
                )
            )
            .alias('rate')
        ).group_by(['code', 'date', 'ebb']).agg(
            pl.col('rate')
            .get(
                pl.col('volume_beside')
                .arg_min()
            ),
            pl.col('volume_beside')
            .min()
        ).group_by(['code', 'date']).agg(
            pl.col('rate')
            .get(
                pl.col('volume_beside')
                .arg_max()
            )
            .alias('RuoShiBanChaoXi')
        )
    )


def cal_ChaoXi(df: pl.DataFrame):
    """
    潮汐因子的计算方式
    :param df:
    :return:
    """
    return (
        df.lazy().sort(by=['code', 'date'])
        .drop_nans()
        .with_columns(
            r_mean_20_ts_pl('QiangShiBanChaoXi'),
            r_std_20_ts_pl('RuoShiBanChaoXi')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (
                    pl.col('QiangShiBanChaoXi_mean')
                    + pl.col('RuoShiBanChaoXi_std')
            ).alias('ChaoXi')
        ).collect()
        .sort(by=['code', 'date'])
    )


# 云开雾散：模糊关联度、模糊金额比和修正模糊价差的20日均值等权合成
### 修正模糊价差的算法为将模糊价差负的部分先除以过去10日负的部分的标准差，再按截面进行调整，除以负的部分的和再乘以总的和。

def cal_MoHuGuanLianDu(df: pl.DataFrame):
    """
    云开雾散的中间因子：模糊关联度
    :param df:
    :return:
    """
    return (
        df.lazy().filter(
            pl.col('close') != 0
        ).with_columns(
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('pct_change')
        ).with_columns(
            pl.col('pct_change')
            .rolling_std(5)
            .over(['code', 'date'])
            .alias('volatility')
        ).with_columns(
            pl.col('volatility')
            .rolling_std(5)
            .over(['code', 'date'])
            .alias('ambiguity')
        ).filter(
            pl.col('time') >= 94000000
        ).group_by(['code', 'date']).agg(
            pl.corr(
                pl.col('ambiguity'),
                pl.col('amount')
            )
            .alias('MoHuGuanLianDu')
        ).collect()
    )


def cal_MoHuJinEBi(df: pl.DataFrame):
    """
    云开雾散的中间因子：模糊金额比
    :param df:
    :return:
    """
    return (
        df.lazy().filter(
            pl.col('close') != 0
        ).with_columns(
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('pct_change')
        ).with_columns(
            pl.col('pct_change')
            .rolling_std(5)
            .over(['code', 'date'])
            .alias('volatility')
        ).with_columns(
            pl.col('volatility')
            .rolling_std(5)
            .over(['code', 'date'])
            .alias('ambiguity')
        ).filter(
            pl.col('time') >= 94000000
        ).group_by(['code', 'date']).agg(
            (
                    pl.col('amount')
                    .filter(
                        pl.col('ambiguity') > pl.col('ambiguity').mean()
                    )
                    .mean() / pl.col('amount').mean()
            ).alias('MoHuJinEBi')
        ).collect()
    )


def cal_MoHuJiaCha(df: pl.DataFrame):
    """
    云开雾散的中间因子：模糊价差
    :param df:
    :return:
    """
    return (
        df.lazy().filter(
            pl.col('close') != 0
        ).with_columns(
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('pct_change')
        ).with_columns(
            pl.col('pct_change')
            .rolling_std(5)
            .over(['code', 'date'])
            .alias('volatility')
        ).with_columns(
            pl.col('volatility')
            .rolling_std(5)
            .over(['code', 'date'])
            .alias('ambiguity')
        ).filter(
            pl.col('time') >= 94000000
        ).group_by(['code', 'date']).agg(
            (
                (
                    pl.col('amount').filter(
                        pl.col('ambiguity') > pl.col('ambiguity').mean()
                    ).mean() / pl.col('amount').mean()
                ) - (
                    pl.col('volume').filter(
                        pl.col('ambiguity') > pl.col('ambiguity').mean()
                    ).mean() / pl.col('volume').mean()
                )
            ).alias('MoHuJiaCha')
        ).collect()
    )


def cal_YunKaiWuSan(df: pl.DataFrame):
    """
    云开雾散计算方法
    :param df:
    :return:
    """
    return (
        df.lazy().sort(by=['code', 'date'])
        .filter(
            (~pl.col('MoHuJiaCha').is_null()) & (~pl.col('MoHuJiaCha').is_nan())
        )
        .with_columns(
            pl.col('MoHuJiaCha')
            .rolling_std(10, min_samples=10, ddof=0)
            .over('code')
            .alias('MoHuJiaCha_std'),
            (
                (
                    pl.col('MoHuJiaCha').filter(
                        pl.col('MoHuJiaCha') < 0
                    )
                ).sum() / pl.col('MoHuJiaCha').sum()
            ).over('date')
            .alias('adjust_coef')
        ).with_columns(
            pl.when(
                (pl.col('MoHuJiaCha') < 0) & (pl.col('MoHuJiaCha_std') != 0)
            ).then(
                (pl.col('MoHuJiaCha') / pl.col('MoHuJiaCha_std'))
                * pl.col('adjust_coef')
            ).otherwise(pl.col('MoHuJiaCha'))
            .alias('XiuZhengMoHuJiaCha')
        ).drop_nans()
        .with_columns(
            r_mean_20_ts_pl('MoHuGuanLianDu'),
            r_mean_20_ts_pl('MoHuJinEBi'),
            r_mean_20_ts_pl('XiuZhengMoHuJiaCha')
        ).with_columns(
            standardize_cs_pl('MoHuGuanLianDu_mean'),
            standardize_cs_pl('MoHuJinEBi_mean'),
            standardize_cs_pl('XiuZhengMoHuJiaCha_mean')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (
                pl.col('MoHuGuanLianDu_mean')
                + pl.col('MoHuJinEBi_mean')
                + pl.col('XiuZhengMoHuJiaCha_mean')
            ).alias('YunKaiWuSan')
        ).collect()
    )


# 勇攀高峰：由攀登因子的20日均值与标准差等权合成

def cal_PanDeng(df: pl.DataFrame):
    """
    勇攀高峰的中间因子：攀登
    :param df:
    :return:
    """
    return (
        df.lazy().filter(
            (pl.col('time') >= 93500000)
            & (pl.col('time') <= 145300000)
        ).with_columns(
            pl.concat_arr(
                pl.col(col)
                .shift(i)
                .over(['code', 'date'])
                for col in ['open', 'high', 'low', 'close']
                for i in range(1, 5)
            )
            .alias('info_arr'),
            (pl.col('close') / pl.col('open') - 1)
            .alias('return')
        ).with_columns(
            (pl.col('info_arr').arr.std() / pl.col('info_arr').arr.mean())
            .alias('better_volatility')
        ).with_columns(
            pl.when(pl.col('better_volatility') != 0)
            .then(pl.col('return') / pl.col('better_volatility'))
            .otherwise(pl.col('return') / (pl.col('better_volatility') + 0.001))
            .alias('ret_to_vol')
        ).filter(
            pl.col('better_volatility') >= (
                    pl.col('better_volatility').mean() + pl.col('better_volatility').std()
            ).over(['code', 'date'])
        ).group_by(['code', 'date']).agg(
            pl.cov(pl.col('ret_to_vol'), pl.col('better_volatility'))
            .alias('PanDeng')
        ).collect()
    )


def cal_YongPanGaoFeng(df: pl.DataFrame):
    """
    勇攀高峰计算方法
    :param df:
    :return:
    """
    return (
        df.lazy().sort(by=['code', 'date'])
        .drop_nans()
        .with_columns(
            r_mean_20_ts_pl('PanDeng'),
            r_std_20_ts_pl('PanDeng')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (pl.col('PanDeng_mean') + pl.col('PanDeng_std'))
            .alias('YongPanGaoFeng')
        ).collect()
    )


# 飞蛾扑火：跳跃度的20日均值与标准差、修正振幅1和修正振幅2的20日均值四者等权合成

def cal_TiaoYueDu(df: pl.DataFrame):
    """
    飞蛾扑火的中间因子：跳跃度
    :param df:
    :return:
    """
    return (
        df.lazy().with_columns(
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('pct_change'),
            pl.col('close')
            .log()
            .diff()
            .over(['code', 'date'])
            .alias('log_change')
        ).filter(
            (pl.col('time') >= 93500000)
            & (pl.col('time') <= 145300000)
        ).with_columns(
            (
                    (pl.col('pct_change') - pl.col('log_change')) * 2
                    - pl.col('log_change').pow(2)
            )
            .alias('taylor_residual')
        ).group_by(['code', 'date']).agg(
            pl.col('taylor_residual')
            .sum()
            .alias('TiaoYueDu')
        ).collect()
    )


def cal_FeiEPuHuo(df: pl.DataFrame):
    """
    飞蛾扑火计算方法。需要使用到量价数据：high/low/close
    :param df:
    :return:
    """
    return (
        df.lazy().sort(by=['code', 'date'])
        .with_columns(
            ((pl.col('high') - pl.col('low')) / pl.col('close').shift(1))
            .over('code')
            .alias('range'),
            (pl.col('high') / pl.col('low').shift(1) - 1)
            .over('code')
            .alias('single_return'),
            (pl.col('high') / pl.col('low').shift(1))
            .over('code')
            .log()
            .alias('log_return')
        ).with_columns(
            pl.when(
                pl.col('TiaoYueDu') > pl.col('TiaoYueDu').mean().over('date')
            )
            .then(pl.col('range'))
            .otherwise(-pl.col('range'))
            .alias('FanZhuanZhenFu'),
            (
                    (pl.col('single_return') - pl.col('log_return')) * 2
                    - pl.col('log_return').pow(2)
            )
            .alias('taylor_residual')
        ).drop_nans()
        .with_columns(
            (r_mean_20_ts_pl('TiaoYueDu') + r_std_20_ts_pl('TiaoYueDu'))
            .alias('TiaoYueDu'),
            r_mean_20_ts_pl('FanZhuanZhenFu'),
            r_mean_20_ts_pl('taylor_residual')
        ).with_columns(
            standardize_cs_pl('TiaoYueDu'),
            standardize_cs_pl('FanZhuanZhenFu_mean'),
            standardize_cs_pl('taylor_residual_mean')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (
                    pl.col('TiaoYueDu')
                    + pl.col('FanZhuanZhenFu_mean')
                    + pl.col('taylor_residual_mean')
            ).alias('FeiEPuHuo')
        ).collect()
    )


# 草木皆兵

def cal_RiBoDongLv(df: pl.DataFrame):
    """草木皆兵的中间因子：日波动率"""
    return (
        df.lazy().sort(by='time').select(
            pl.col('code'),
            pl.col('date'),
            pl.col('close')
            .pct_change()
            .over('code')
            .alias('min_return')
        ).drop_nans()
        .group_by(['code', 'date']).agg(
            pl.col('min_return').std()
            .alias('RiBoDongLv')
        ).collect()
    )


def cal_CaoMuJieBing(df: pl.DataFrame):
    """草木皆兵因子计算方法"""
    return (
        df.lazy().with_columns(
            pl.col('close')
            .pct_change()
            .over('code')
            .alias('daily_return')
        ).with_columns(
            (
                (
                    pl.col('daily_return') * pl.col('cmc')
                ).sum() / (pl.col('cmc').sum())
            ).over('date')
            .alias('bench_return')
        ).with_columns(
            (
                (
                    pl.col('daily_return').abs() - pl.col('bench_return').abs()
                ) / (
                    pl.col('daily_return').abs() + pl.col('bench_return').abs() + 0.1
                )
            ).alias('JingKongDu')
        ).with_columns(
            (
                pl.col('JingKongDu')
                - pl.col('JingKongDu').rolling_mean(2, min_samples=2).shift(1)
            )
            .over('code')
            .alias('JingKongDu_ShuaiJian')
        ).with_columns(
            pl.when(pl.col('JingKongDu_ShuaiJian') > 0)
            .then(pl.col('JingKongDu_ShuaiJian'))
            .otherwise(0)
            .alias('JingKongDu_ShuaiJian')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (
                pl.col('RiBoDongLv') * pl.col('daily_return') * (
                    pl.col('JingKongDu') + pl.col('JingKongDu_ShuaiJian')
                )
            ).alias('CaoMuJieBing')
        ).collect()
    )


# 水中行舟：孤雁出群的20日均值与标准差和随波逐流三者等权合成

def cal_GuYanChuQun(df: pl.DataFrame):
    """
    水中行舟的中间因子：孤雁出群
    :param df:
    :return:
    """
    pivot_df = (
        df.filter(
            pl.col('close') != 0
        ).with_columns(
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('pct_change')
        ).with_columns(
            pl.col('pct_change')
            .std()
            .over(['date', 'time'])
            .alias('differentiation')
        ).filter(
            pl.col('differentiation') < pl.col('differentiation').mean().over('date')
        ).select(
            'code', 'time', 'amount', 'date'
        ).pivot(on='time', index='code', values='amount')
    )
    return (
        pivot_df.select(
            pl.col('code'),
            pl.lit(df['date'].first())
            .alias('date'),
            pl.Series(
                torch.corrcoef(
                    pivot_df.select(
                        pl.all().exclude('code')
                    ).to_torch()
                ).abs()
                .nanmean(dim=1, keepdim=True)
            )
            .arr.first()
            .fill_nan(None)
            .alias('GuYanChuQun')
        )
    )


def cal_GaoDiECha(df: pl.DataFrame, daily_pv: pl.DataFrame):
    """
    水中行舟的中间因子：随波逐流的中间因子：高低额差
    :param df:
    :param daily_pv:
    :return:
    """
    return (
        pl.concat(
            items=[
                df,
                daily_pv.select(
                    pl.col('Stkcd')
                    .cast(pl.String)
                    .alias('code'),
                    pl.col('Trddt')
                    .str.to_date(format='%Y-%m-%d')
                    .alias('date'),
                    pl.col('Dsmvosd')
                    .alias('mv'),
                    (pl.col('Clsprc') / pl.col('Opnprc') - 1)
                    .rolling_mean(20)
                    .over('Stkcd')
                    .alias('reasonable_return')  # 合理收益率
                ).filter(
                    pl.col('mv') != 0
                )
            ], how='align_left'
        ).lazy().filter(
            pl.col('close') != 0
        ).with_columns(
            (pl.col('close') / pl.col('close').first() - 1)
            .over(['code', 'date'])
            .alias('intraday_return')
        ).with_columns(
            pl.when(pl.col('intraday_return') > pl.col('reasonable_return'))
            .then(1)
            .otherwise(0)
            .alias('high_moment'),
            pl.when(pl.col('intraday_return') > pl.col('reasonable_return'))
            .then(0)
            .otherwise(1)
            .alias('low_moment')
        ).group_by(['code', 'date']).agg(
            (pl.col('high_moment') * pl.col('amount'))
            .sum()
            .alias('high_amount'),
            (pl.col('low_moment') * pl.col('amount'))
            .sum()
            .alias('low_amount'),
            (pl.col('mv') * 1000).last()
        ).select(
            pl.col('code'),
            pl.col('date'),
            ((pl.col('high_amount') - pl.col('low_amount')) / pl.col('mv'))
            .alias('GaoDiECha')
        ).collect()
    )


def cal_SuiBoZhuLiu(df: pl.DataFrame):
    """
    水中行舟的中间因子：随波逐流
    :param df:
    :return:
    """
    pivot_data = (
        df.select(
            'date', 'code', 'GaoDiECha'
        ).sort(by=['date', 'code'])
        .pivot(index='date', on='code', values='GaoDiECha')
        .to_pandas()
    )

    def _single_corr_cal(df_rolling: pd.DataFrame, i):
        return (
            df_rolling.loc[i - 20: i].set_index('date')
            .dropna(axis=1)
            .corr(method='spearman').abs().mean()
            .rename(index=pivot_data.loc[i, 'date']).to_frame().T
        )

    valid_results = []
    if pivot_data.shape[0] >= 20:
        results = Parallel(n_jobs=-1)(
            delayed(_single_corr_cal)(
                pivot_data,
                i
            )
            for i in tqdm(range(20, pivot_data.shape[0]), desc='Processing')
        )
        valid_results = [r for r in results if r is not None]
    if len(valid_results) == 0:
        return None
    else:
        return (
            pl.from_pandas(
                pd.concat(valid_results).unstack(level=1).reset_index()
            ).select(
                pl.col('level_0')
                .alias('code'),
                pl.col('level_1')
                .cast(pl.Date)
                .alias('date'),
                pl.col('0')
                .alias('SuiBoZhuLiu')
            ).drop_nulls()
            .sort(by=['code', 'date'])
        )


def cal_ShuiZhongXingZhou(df: pl.DataFrame):
    """
    水中行舟计算方法
    :param df:
    :return:
    """
    return (
        df.lazy().sort(by=['code', 'date'])
        .drop_nans()
        .with_columns(
            (r_mean_20_ts_pl('GuYanChuQun') + r_std_20_ts_pl('GuYanChuQun'))
            .alias('GuYanChuQun')
        ).with_columns(
            standardize_cs_pl('GuYanChuQun'),
            standardize_cs_pl('SuiBoZhuLiu')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (pl.col('GuYanChuQun') + pl.col('SuiBoZhuLiu'))
            .alias('ShuiZhongXingZhou')
        ).collect()
    )


# 花隐林间：朝没晨雾和午避古木的20日均值与该股票与当期截面所有股票过去20天的夜眠霜路的相关系数的绝对值等权合成

def _HuaYinLinJian_preprocess(df: pl.DataFrame) -> pl.DataFrame:
    """花隐林间中间因子统一的预处理方式"""
    return (
        df.filter(
            pl.col('close') != 0
        ).with_columns(
            pl.col('close')
            .pct_change()
            .over(['code', 'date'])
            .alias('pct_change'),
            pl.col('volume')
            .diff()
            .over(['code', 'date'])
            .alias('vol_diff')
        ).with_columns(
            pl.col('vol_diff')
            .shift(i)
            .over(['code', 'date'])
            .alias(f'vol_diff_shift_{i}') for i in range(0, 6)
        ).filter(
            (pl.col('time') > 93700000) & (pl.col('time') <= 145300000)
        ).drop_nulls()
    )


def cal_ZhaoMoChenWu(df: pl.DataFrame):
    """
    花隐林间的中间因子：朝没晨雾
    :param df:
    :return:
    """
    return pl.from_pandas(
        _HuaYinLinJian_preprocess(df)
        .to_pandas()
        .groupby(['code', 'date']).apply(
            lambda g: sm.OLS(
                g['pct_change'], sm.add_constant(
                    g[[f'vol_diff_shift_{i}' for i in range(0, 6)]]
                )
            ).fit().tvalues,
            include_groups=False
        ).reset_index(level=['code', 'date'])
    ).select(
        pl.col('code'),
        pl.col('date')
        .cast(pl.Date),
        pl.concat_arr(
            pl.col(f'vol_diff_shift_{i}') for i in range(1, 6)
        ).arr.std()
        .alias('ZhaoMoChenWu')
    )


def cal_WuBiGuMu(df: pl.DataFrame):
    """
    花隐林间的中间因子：午避古木
    :param df:
    :return:
    """
    return pl.from_pandas(
        _HuaYinLinJian_preprocess(df)
        .to_pandas()
        .groupby(['code', 'date']).apply(
            lambda g: sm.OLS(
                g['pct_change'], sm.add_constant(
                    g[[f'vol_diff_shift_{i}' for i in range(0, 6)]]
                )
            ).fit(),
            include_groups=False
        ).apply(
            lambda m: pd.Series(
                [m.tvalues.iloc[0], m.fvalue],
                index=['t_intercept', 'f']
            )
        ).reset_index(level=['code', 'date'])
    ).select(
        pl.col('code'),
        pl.col('date')
        .cast(pl.Date),
        pl.when(pl.col('f') < pl.col('f').mean())
        .then(pl.col('t_intercept').abs() * -1)
        .otherwise(pl.col('t_intercept').abs())
        .alias('WuBiGuMu')
    )


def cal_YeMianShuangLu_t_intercept(df: pl.DataFrame):
    """
    花隐林间的中间因子的中间因子：夜眠霜路的中间因子：t_intercept
    :param df:
    :return:
    """
    return pl.from_pandas(
        _HuaYinLinJian_preprocess(df)
        .to_pandas()
        .groupby(['code', 'date']).apply(
            lambda g: sm.OLS(
                g['pct_change'], sm.add_constant(
                    g[[f'vol_diff_shift_{i}' for i in range(0, 6)]]
                )
            ).fit().tvalues.iloc[0],
            include_groups=False
        ).reset_index(level=['code', 'date'])
    ).select(
        pl.col('code'),
        pl.col('date')
        .cast(pl.Date),
        pl.col('0')
        .alias('YeMianShuangLu_t_intercept')
    )


def cal_YeMianShuangLu(df: pl.DataFrame):
    """
    花隐林间的中间因子：夜眠霜路
    :param df:
    :return:
    """
    pivot_data = (
        df.select(
            'date', 'code', 'YeMianShuangLu_t_intercept'
        ).sort(by=['date', 'code'])
        .pivot(index='date', on='code', values='YeMianShuangLu_t_intercept')
        .to_pandas()
    )
    from tqdm import tqdm
    from joblib import Parallel, delayed
    def _single_corr_cal(df_rolling: pd.DataFrame, i):
        return (
            df_rolling.loc[i - 20: i].set_index('date')
            .corr().abs().mean()
            .rename(index=pivot_data.loc[i, 'date']).to_frame().T
        )

    valid_results = []
    if pivot_data.shape[0] >= 20:
        results = Parallel(n_jobs=-1)(
            delayed(_single_corr_cal)(
                pivot_data,
                i
            )
            for i in tqdm(range(20, pivot_data.shape[0]), desc='Processing')
        )
        valid_results = [r for r in results if r is not None]
    if len(valid_results) == 0:
        return None
    else:
        return (
            pl.from_pandas(
                pd.concat(valid_results).unstack(level=1).reset_index()
            ).select(
                pl.col('level_0')
                .alias('code'),
                pl.col('level_1')
                .cast(pl.Date)
                .alias('date'),
                pl.col('0')
                .alias('YeMianShuangLu')
            )
        )


def cal_HuaYinLinJian(df: pl.DataFrame):
    """
    花隐林间计算方法。
    :param df:
    :return:
    """
    return (
        df.lazy().sort(by=['code', 'date'])
        .drop_nans()
        .select(
            pl.col('code'),
            pl.col('date'),
            (
                (
                    pl.col('ZhaoMoChenWu')
                    .rolling_mean(20, min_samples=20)
                ) + (
                    pl.col('WuBiGuMu')
                    .rolling_mean(20, min_samples=20)
                ) + (
                    pl.col('YeMianShuangLu')
                    .fill_null(0)
                )
            ).over('code')
            .alias('HuaYinLinJian')
        ).collect()
    )


# 待著而救：由跟随系数的20日均值与标准差等权合成

def cal_GenSuiXiShu(df: pl.DataFrame):
    """
    待著而救的中间因子：跟随系数
    :param df:
    :return:
    """
    return (
        df.lazy().filter(
            pl.col('time') >= 94500000
        ).with_columns(
            pl.when(pl.col('volume') >= pl.col('volume').top_k(10).min())
            .then(pl.col('time'))
            .otherwise(None)
            .over('code')
            .alias('high_volume_moment')
        ).with_columns(
            (pl.col('high_volume_moment') / pl.col('high_volume_moment'))
            .alias('advantage_moment'),
            pl.col('high_volume_moment')
            .forward_fill()
            .over('code')
            .diff()
            .alias('moment_diff')
        ).with_columns(
            pl.when(pl.col('moment_diff') < 500000)
            .then(None)
            .otherwise(pl.col('advantage_moment'))
            .fill_null(0)
            .alias('advantage_moment'),
            pl.lit(0).alias('follow_moment')
        ).with_columns(
            (
                pl.col('follow_moment')
                + pl.col('advantage_moment').shift(0)
                + pl.col('advantage_moment').shift(1)
                + pl.col('advantage_moment').shift(2)
                + pl.col('advantage_moment').shift(3)
                + pl.col('advantage_moment').shift(4)
            )
            .over('code')
        ).group_by(['code', 'date']).agg(
            (
                (
                    (pl.col('follow_moment') * pl.col('volume'))
                    .sum()
                ) / (
                    pl.col('volume').sum()
                )
            ).alias('GenSuiXiShu')
        ).collect()
    )


def cal_DaiZhuErJiu(df: pl.DataFrame):
    """待著而救因子计算方法"""
    return (
        df.lazy().drop_nans()
        .with_columns(
            r_mean_20_ts_pl('GenSuiXiShu'),
            r_std_20_ts_pl('GenSuiXiShu')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (pl.col('GenSuiXiShu_mean') + pl.col('GenSuiXiShu_std'))
            .alias('DaiZhuErJiu')
        ).collect()
    )


# 多空博弈：由均值距离化后的成交量博弈-收益率、成交量博弈-日内相对位置和振幅博弈的20日均值与标准差等权合成

def cal_ChengJiaoLiangBoYi_ShouYiLv(df: pl.DataFrame):
    """多空博弈的中间因子：成交量博弈-收益率"""
    return (
        df.lazy().with_columns(
            (pl.col('close') / pl.col('open').shift(4) - 1)
            .over('code')
            .alias('return_past_5_min')
        ).filter(
            (pl.col('time') > 93500000) & (pl.col('time') < 145700000)
        ).with_columns(
            pl.col('volume')
            .sort_by('return_past_5_min', descending=False)
            .over('code')
            .alias('volume_ascend'),
            pl.col('volume')
            .sort_by('return_past_5_min', descending=True)
            .over('code')
            .alias('volume_descend')
        ).group_by(['code', 'date']).agg(
            (pl.col('volume_ascend').cum_sum() - pl.col('volume_descend').cum_sum())
            .sum()
            .alias('ChengJiaoLiangBoYi_ShouYiLv')
        ).collect()
    )


def cal_ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi(df: pl.DataFrame):
    """多空博弈的中间因子：成交量博弈-日内相对位置"""
    return (
        df.lazy().with_columns(
            pl.col('low')
            .cum_min()
            .over('code')
            .alias('former_low'),
            pl.col('high')
            .cum_max()
            .over('code')
            .alias('former_high')
        ).with_columns(
            (
                (
                    (
                        pl.col('close') / pl.col('former_high')
                    ) + (
                        pl.col('close') / pl.col('former_low')
                    )
                ) / 2 - 1
            ).alias('position')
        ).filter(
            (pl.col('time') > 93500000) & (pl.col('time') < 145700000)
        ).with_columns(
            pl.col('volume')
            .sort_by('position', descending=False)
            .over('code')
            .alias('volume_ascend'),
            pl.col('volume')
            .sort_by('position', descending=True)
            .over('code')
            .alias('volume_descend')
        ).group_by(['code', 'date']).agg(
            (pl.col('volume_ascend').cum_sum() - pl.col('volume_descend').cum_sum())
            .sum()
            .alias('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi')
        ).collect()
    )


def cal_ZhenFuBoYi(df: pl.DataFrame):
    """多空博弈的中间因子：振幅博弈"""
    return (
        df.lazy().with_columns(
            (pl.col('close') / pl.col('open').shift(4) - 1)
            .over('code')
            .alias('return_past_5_min'),
            ((pl.col('high') - pl.col('low')) / pl.col('close'))
            .alias('range')
        ).filter(
            (pl.col('time') > 93500000) & (pl.col('time') < 145700000)
        ).with_columns(
            pl.col('range')
            .sort_by('return_past_5_min', descending=False)
            .over('code')
            .alias('range_ascend'),
            pl.col('range')
            .sort_by('return_past_5_min', descending=True)
            .over('code')
            .alias('range_descend')
        ).group_by(['code', 'date']).agg(
            (pl.col('range_ascend').cum_sum() - pl.col('range_descend').cum_sum())
            .sum()
            .alias('ZhenFuBoYi')
        ).collect()
    )


def cal_DuoKongBoYi(df: pl.DataFrame):
    """多空博弈计算方法"""
    return (
        df.lazy().drop_nans()
        .with_columns(
            standardize_cs_pl('ChengJiaoLiangBoYi_ShouYiLv'),
            standardize_cs_pl('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi')
        ).with_columns(
            (
                pl.col('ChengJiaoLiangBoYi_ShouYiLv')
                - pl.col('ChengJiaoLiangBoYi_ShouYiLv').mean()
            ).abs()
            .over('date')
            .alias('ChengJiaoLiangBoYi_ShouYiLv'),
            (
                pl.col('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi')
                - pl.col('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi').mean()
            ).abs()
            .over('date')
            .alias('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi'),
            (
                    pl.col('ZhenFuBoYi')
                    - pl.col('ZhenFuBoYi').mean()
            ).abs()
            .over('date')
            .alias('ZhenFuBoYi')
        ).with_columns(
            (
                r_mean_20_ts_pl('ChengJiaoLiangBoYi_ShouYiLv')
                + r_std_20_ts_pl('ChengJiaoLiangBoYi_ShouYiLv')
            ).alias('ChengJiaoLiangBoYi_ShouYiLv'),
            (
                r_mean_20_ts_pl('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi')
                + r_std_20_ts_pl('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi')
            ).alias('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi'),
            (
                r_mean_20_ts_pl('ZhenFuBoYi')
                + r_std_20_ts_pl('ZhenFuBoYi')
            ).alias('ZhenFuBoYi')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (
                pl.col('ChengJiaoLiangBoYi_ShouYiLv')
                + pl.col('ChengJiaoLiangBoYi_RiNeiXiangDuiWeiZhi')
                + pl.col('ZhenFuBoYi') * 2
            ).alias('DuoKongBoYi')
        ).collect()
    )


# 协同效应：成交量协同和协同价差的20日均值与标准差

def cal_ChengJiaoLiangXieTong(df: pl.DataFrame):
    """协同效应的中间因子：成交量协同"""
    return (
        df.lazy().with_columns(
            pl.concat_arr(
                pl.col('high'), pl.col('low'),
                pl.col('open'), pl.col('close'),
                pl.col('high').shift(1), pl.col('low').shift(1),
                pl.col('open').shift(1), pl.col('close').shift(1),
                pl.col('high').shift(2), pl.col('low').shift(2),
                pl.col('open').shift(2), pl.col('close').shift(2),
                pl.col('high').shift(3), pl.col('low').shift(3),
                pl.col('open').shift(3), pl.col('close').shift(3),
                pl.col('high').shift(4), pl.col('low').shift(4),
                pl.col('open').shift(4), pl.col('close').shift(4)
            ).alias('past_20_variable')
        ).filter(
            pl.col('time') > 93500000
        ).with_columns(
            pl.col('past_20_variable')
            .arr.mean()
            .alias('past_mean'),
            pl.col('past_20_variable')
            .arr.std()
            .alias('past_std')
        ).with_columns(
            (pl.col('past_mean') + pl.col('past_std'))
            .alias('upper_track'),
            (pl.col('past_mean') - pl.col('past_std'))
            .alias('lower_track')
        ).with_columns(
            pl.when(
                pl.col('close') > pl.col('upper_track')
            ).then(
                1
            ).otherwise(
                pl.when(
                    pl.col('close') < pl.col('lower_track')
                )
                .then(-1)
                .otherwise(0)
            ).alias('co_state'),
            pl.when(pl.col('volume').sum() != 0)
            .then(pl.col('volume') / pl.col('volume').sum())
            .otherwise(0)
            .over('code')
            .alias('volume_d')
        ).with_columns(
            pl.col('volume_d')
            .sum()
            .over(['time', 'co_state'])
            .alias('co_volume_d')
        ).group_by(['code', 'date']).agg(
            pl.corr(
                pl.col('volume_d'),
                (pl.col('co_volume_d') - pl.col('volume_d'))
            ).fill_nan(0)
            .alias('ChengJiaoLiangXieTong')
        ).collect()
    )


def cal_XieTongJiaCha(df: pl.DataFrame):
    """协同效应的中间因子：协同价差"""
    df = (
        df.filter(
            (pl.col('close') != 0) & (pl.col('open') != 0)
        ).with_columns(
            pl.col('close')
            .pct_change()
            .over('code')
            .alias('pct_change'),
            (pl.col('close').shift(1) / pl.col('open').shift(4) - 1)
            .over('code')
            .alias('pct_change_5'),
            (pl.col('volume').rolling_sum(5) - pl.col('volume'))
            .over('code')
            .alias('volume_5')
        ).with_columns(
            pl.col('pct_change')
            .sign()
            .alias('sign_1'),
            (pl.col('pct_change_5').sign() * pl.col('pct_change').sign())
            .alias('sign_2'),
            (pl.col('volume') - pl.col('volume_5'))
            .sign()
            .alias('sign_3')
        ).filter((pl.col('time') >= 93500000) & (pl.col('time') < 145700000))
        .sort(by='code')
    )
    pivot_info = df.pivot(on='time', index='code', values='sign_1').fill_null(0)
    pivot_1 = (
        pivot_info.select(
            pl.all().exclude('code')
        ).to_torch()
    )
    pivot_2 = (
        df.pivot(on='time', index='code', values='sign_2')
        .fill_null(0)
        .select(
            pl.all().exclude('code')
        ).to_torch()
    )
    pivot_3 = (
        df.pivot(on='time', index='code', values='sign_3')
        .fill_null(0)
        .select(
            pl.all().exclude('code')
        ).to_torch()
    )
    co_table = (
        pl.from_torch(
            (
                torch.mm(pivot_1, pivot_1.T)
                + torch.mm(pivot_2, pivot_2.T)
                + torch.mm(pivot_3, pivot_3.T)
            )
        ).select(
            pl.all().is_in(pl.all().top_k(31).implode())
            .cast(pl.Float64)
        ).to_torch()
    )
    df = (
        df.group_by(['code', 'date']).agg(
            (pl.col('close').last() / pl.col('close').first() - 1)
            .alias('return')
        ).sort(by='code')
    )
    pct_change = (
        df.select('return')
        .to_torch()
    )
    return df.select(
        pl.all().exclude('return'),
        pl.Series(
            pct_change - (
                torch.mm(
                    co_table,
                    pct_change
                ) - pct_change
            ) / 30
        ).arr.first()
        .alias('XieTongJiaCha')
    )


def cal_XieTongXiaoYing(df: pl.DataFrame):
    """协同效应计算方法"""
    return (
        df.lazy().drop_nans()
        .with_columns(
            (
                r_mean_20_ts_pl('ChengJiaoLiangXieTong')
                + r_std_20_ts_pl('ChengJiaoLiangXieTong')
            ).alias('ChengJiaoLiangXieTong'),
            (
                r_mean_20_ts_pl('XieTongJiaCha')
                + r_std_20_ts_pl('XieTongJiaCha')
            ).alias('XieTongJiaCha')
        ).with_columns(
            standardize_cs_pl('ChengJiaoLiangXieTong'),
            standardize_cs_pl('XieTongJiaCha')
        ).select(
            pl.col('code'),
            pl.col('date'),
            (
                pl.col('ChengJiaoLiangXieTong')
                + pl.col('XieTongJiaCha')
            ).alias('XieTongXiaoYing')
        ).collect()
    )
