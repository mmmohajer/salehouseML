import pandas as pd
import numpy as np

class FeatureHandler():

	def __init__(self, df, effective_cols = None, coeff_list = None):
		self.df = df
		if effective_cols:
			self.effective_cols = effective_cols
		else:
			self.effective_cols = df.columns
		if coeff_list:
			self.coeff_list = coeff_list
		else:
			self.coeff_list = False
		self.missing_info = self.cols_with_missing_info()
		self.numeric_cols = self.numeric_col_finder()
		self.categoric_cols = self.categoric_col_finder()
		self.scaled_df = self.numeric_col_scaler()

	def cols_with_missing_info(self):
		col_name = []
		col_type = []
		total_cat = []
		num_null = []
		null_percentage = []
		sample_value = []
		for col in self.df.columns:
				if self.df[col].isnull().mean() > 0:
					col_name.append(col)
					col_type.append(self.df[col].dtype.name.capitalize())
					total_cat.append(len(self.df[col].value_counts()))
					num_null.append(self.df[col].isnull().sum())
					null_percentage.append(np.round(self.df[col].isnull().mean() * 100, 2))
					sample_value.append(self.df[col].value_counts().sort_values(ascending = False).reset_index()["index"][0])
		newdf = pd.DataFrame({"Var Name": col_name,
							  "Data Type": col_type,
							  "Tot Cat": total_cat,
							  "Num of Missing Val": num_null,
							  "Null Percentage (%)": null_percentage,
							  "Sample Val": sample_value})
		newdf.sort_values(by = ["Data Type", "Null Percentage (%)"], 
						  ascending = [False, False], 
						  inplace = True)
		newdf.reset_index(inplace = True)
		newdf.drop("index", axis = 1, inplace = True)
		return newdf

	def numeric_col_finder(self):
		cols = self.df.columns
		numeric_cols = []
		for col in cols:
			if self.df[col].dtype.name != 'object':
				numeric_cols.append(col)
		return numeric_cols

	def categoric_col_finder(self):
		cols = self.df.columns
		categoric_cols = []
		for col in cols:
			if self.df[col].dtype.name == 'object':
				categoric_cols.append(col)
		return categoric_cols

	def numeric_col_scaler(self, numeric_cols = None):
		if numeric_cols:
			numeric_cols = numeric_cols
		else:
			numeric_cols = self.numeric_cols
		newdf = pd.DataFrame()
		for col in numeric_cols:
			max_val_col = np.max(np.absolute(self.df[col]))
			newdf[col] = self.df[col] / max_val_col
		return newdf

	def diff_vec_norm(self, idx1, idx2):
		vec_len = 0
		if self.coeff_list:
			j =0
		for col in self.effective_cols:
			missing_index_list = self.df[self.df[col].isnull()].index.tolist()
			if self.coeff_list:
				if idx1 not in missing_index_list and idx2 not in missing_index_list:
					if col in self.numeric_cols:
						vec_len = vec_len + ((self.scaled_df[col].loc[idx2] - self.scaled_df[col].loc[idx1]) ** 2) * self.coeff_list[j]
					else:
						if self.df[col].loc[idx2] != self.df[col].loc[idx1]:
							vec_len = vec_len + self.coeff_list[j]
				else:
					vec_len = vec_len + self.coeff_list[j]
				j += 1
			else:
				if idx1 not in missing_index_list and idx2 not in missing_index_list:
					if col in self.numeric_cols:
						vec_len = vec_len + (self.scaled_df[col].loc[idx2] - self.scaled_df[col].loc[idx1]) ** 2
					else:
						if self.df[col].loc[idx2] != self.df[col].loc[idx1]:
							vec_len = vec_len + 1
				else:
					vec_len = vec_len + 1
		vec_len = np.sqrt(vec_len)
		return vec_len

	def similar_index_finder(self, col, missing_index, num_comprised_rows = 10, num_eff_items = 4, rand_seed = 42):
		missing_index_list = self.df[self.df[col].isnull()].index.tolist()
		all_rows = self.df.index
		filled_rows = np.array(list(set(all_rows) - set(missing_index_list)))
		np.random.seed(seed = rand_seed)
		selected_row = np.random.randint(len(filled_rows), size = num_comprised_rows)
		vec_len_list = []
		for item in range(0, num_comprised_rows):
			vec_len = self.diff_vec_norm(missing_index, filled_rows[selected_row[item]])
			vec_len_list.append(vec_len)
		vec_len_oredered_idx = np.argsort(vec_len_list)
		eff_index_list = []
		for item in range(num_eff_items):
			eff_index_list.append(selected_row[vec_len_oredered_idx[item]])
		eff_values = np.array(self.df[col].loc[filled_rows[eff_index_list]])
		return eff_values

	def num_filling_val_finder(self, col, eff_values, strategy = 'median', num_is_int = True):
		if strategy == 'mean':
			filling_val = np.mean(eff_values)
		else:
			filling_val = np.median(eff_values)
		if num_is_int:
			filling_val = np.round(filling_val)
		return filling_val

	def cat_filling_val_finder(df, col, eff_values):
		created_df = pd.DataFrame({'Filling_Values': eff_values})
		filling_val = created_df['Filling_Values'].mode()[0]
		return filling_val

	def knn_num_imputer(self, num_is_int, strategy, num_comprised_rows = 10, num_eff_items = 5, rand_seed = 42,
						to_be_filled_cols = None):
		pd.options.mode.chained_assignment = None
		if to_be_filled_cols:
			to_be_filled_cols = to_be_filled_cols
		else:
			to_be_filled_cols = self.missing_info["Var Name"]
		newdf = self.df.copy()
		j = 0
		for col in to_be_filled_cols:
			missing_index_list = self.df[self.df[col].isnull()].index.tolist()
			for missing_index in missing_index_list:
				eff_values = self.similar_index_finder(col, missing_index, num_comprised_rows, num_eff_items, rand_seed)
				filling_val = self.num_filling_val_finder(col, eff_values, strategy[j], num_is_int[j])
				newdf[col].loc[missing_index] = filling_val
			j += 1
		return newdf

	def knn_cat_imputer(self, num_comprised_rows = 10, num_eff_items = 5, rand_seed = 42, to_be_filled_cols = None):
		pd.options.mode.chained_assignment = None
		if to_be_filled_cols:
			to_be_filled_cols = to_be_filled_cols
		else:
			to_be_filled_cols = self.missing_info["Var Name"]
		newdf = self.df.copy()
		j = 0
		for col in to_be_filled_cols:
			missing_index_list = self.df[self.df[col].isnull()].index.tolist()
			for missing_index in missing_index_list:
				eff_values = self.similar_index_finder(col, missing_index, num_comprised_rows, num_eff_items, rand_seed)
				filling_val = self.cat_filling_val_finder(col, eff_values)
				newdf[col].loc[missing_index] = filling_val
			j += 1
		return newdf

	def simple_num_imputer(self, cols, strategy, num_is_int, rand_seed = 42):
		i = 0
		strategies = ["mean", "median", "end_tail", "end_tail_norm", "mean_indicator", "median_indicator"]
		new_df = self.df.copy()
		for col in cols:
			if strategy[i] != "random":
				if strategy[i] == "mean":
					filling_val = new_df[col].mean()
				elif strategy[i] == "median": 
					filling_val = new_df[col].median()
				elif strategy[i] == "end_tail":
					iqr = new_df[col].quantile(0.75) - new_df[col].quantile(0.25)
					filling_val = new_df[col].mean() + 3 * iqr
				elif strategy[i] == "end_tail_norm":
					filling_val = new_df[col].mean() + 3 * new_df[col].std()
				elif strategy[i] == "mean_indicator":
					newcol = col + "_NA"
					new_df[newcol] = np.where(new_df[col].isnull(), 1, 0)
					filling_val = new_df[col].mean()
				elif strategy[i] == "median_indicator":
					newcol = col + "_NA"
					new_df[newcol] = np.where(new_df[col].isnull(), 1, 0)
					filling_val = new_df[col].median()
				else:
					print(f"For performing imputation on {col} you must choose another method!")
				if strategy[i] in strategies:
					if num_is_int[i]:
						filling_val = np.round(filling_val)
					new_df[col].fillna(filling_val, inplace = True)
			else:
				filled_val = np.array(new_df[col].dropna())
				missed_len = new_df[col].isnull().sum()
				filled_len = len(new_df) - missed_len
				np.random.seed(rand_seed)
				rand_list = np.random.randint(low = 0, high = filled_len, size = missed_len)
				imputed_list = np.array(filled_val[rand_list])
				new_df.loc[new_df[col].isnull(), col] = imputed_list
			i += 1
		return new_df

	def simple_cat_imputer(self, cols, strategy, rand_seed = 42):
		i = 0
		new_df = self.df.copy()
		for col in cols:
			if strategy[i] == "mode":
				mydf = pd.DataFrame(new_df[col].value_counts().sort_values(ascending=False))
				mydf.reset_index(inplace = True)
				mode_val = mydf["index"][0]
				new_df[col].fillna(mode_val, inplace = True)
			elif strategy[i] == "missing": 
				new_df[col].fillna("Missing", inplace = True)
			elif strategy[i] == "random":
				filled_val = np.array(new_df[col].dropna())
				missed_len = new_df[col].isnull().sum()
				filled_len = len(new_df) - missed_len
				np.random.seed(rand_seed)
				rand_list = np.random.randint(low = 0, high = filled_len, size = missed_len)
				imputed_list = np.array(filled_val[rand_list])
				new_df.loc[new_df[col].isnull(), col] = imputed_list
			elif strategy[i] == "mode_indicator":
				newcol = col + "_NA"
				new_df[newcol] = np.where(new_df[col].isnull(), 1, 0)
				mydf = pd.DataFrame(new_df[col].value_counts().sort_values(ascending=False))
				mydf.reset_index(inplace = True)
				mode_val = mydf["index"][0]
				new_df[col].fillna(mode_val, inplace = True)
			else:
				print(f"For performing imputation on {col} you must choose another method!")
			i += 1
		return new_df

	def test_set_simple_num_imputer(self, df_train, cols, strategy, num_is_int, rand_seed = 42):
		df_test = self.df.copy()
		i = 0
		for col in cols:
			test_missing_index_list = df_test[df_test[col].isnull()].index.tolist()
			df_test_missing = df_test.loc[test_missing_index_list]
			df_test.dropna(subset=[col], inplace = True)
			self.df = pd.concat([df_train, df_test_missing])
			new_df = self.simple_num_imputer([col], [strategy[i]], [num_is_int[i]], rand_seed)
			df_test_imputed = new_df.loc[test_missing_index_list]
			df_test = pd.concat([df_test, df_test_imputed])
			i += 1
		return df_test

	def test_set_simple_cat_imputer(self, df_train, cols, strategy, rand_seed = 42):
		df_test = self.df.copy()
		i = 0
		for col in cols:
			test_missing_index_list = df_test[df_test[col].isnull()].index.tolist()
			df_test_missing = df_test.loc[test_missing_index_list]
			df_test.dropna(subset=[col], inplace = True)
			self.df = pd.concat([df_train, df_test_missing])
			new_df = self.simple_cat_imputer([col], [strategy[i]], rand_seed)
			df_test_imputed = new_df.loc[test_missing_index_list]
			df_test = pd.concat([df_test, df_test_imputed])
			i += 1
		return df_test

	def test_set_knn_num_imputer(self, df_train, cols, strategy, num_is_int, num_comprised_rows, num_eff_items, rand_seed = 42):
		df_test = self.df.copy()
		i = 0
		for col in cols:
			test_missing_index_list = df_test[df_test[col].isnull()].index.tolist()
			df_test_missing = df_test.loc[test_missing_index_list]
			df_test.dropna(subset=[col], inplace = True)
			self.df = pd.concat([df_train, df_test_missing])
			self.scaled_df = self.numeric_col_scaler()
			new_df = self.knn_num_imputer([num_is_int[i]], [strategy[i]], num_comprised_rows, num_eff_items, rand_seed, [col])
			df_test_imputed = new_df.loc[test_missing_index_list]
			df_test = pd.concat([df_test, df_test_imputed])
			i += 1
		return df_test

	def test_set_knn_cat_imputer(self, df_train, cols, num_comprised_rows, num_eff_items, rand_seed = 42):
		df_test = self.df.copy()
		i = 0
		for col in cols:
			test_missing_index_list = df_test[df_test[col].isnull()].index.tolist()
			df_test_missing = df_test.loc[test_missing_index_list]
			df_test.dropna(subset=[col], inplace = True)
			self.df = pd.concat([df_train, df_test_missing])
			self.scaled_df = self.numeric_col_scaler()
			new_df = self.knn_cat_imputer(num_comprised_rows, num_eff_items, rand_seed, [col])
			df_test_imputed = new_df.loc[test_missing_index_list]
			df_test = pd.concat([df_test, df_test_imputed])
			i += 1
		return df_test